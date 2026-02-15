import traceback
import config


def _bbox_fully_inside(bbox, roi):
	x1, y1, x2, y2 = bbox
	# Controlled tolerance: allow small spatial margin and/or high overlap ratio
	# Reasonable defaults tuned for jitter: larger margin and slightly lower ratio
	MARGIN = getattr(config, 'PLATE_ROI_MARGIN', 12)  # pixels tolerance (8-15 recommended)
	RATIO_THRESH = getattr(config, 'PLATE_ROI_INSIDE_RATIO', 0.85)  # optional ratio-based fallback

	roi_top = roi.get('y1', -9999)
	roi_bottom = roi.get('y2', 99999)

	# Strict-with-margin containment
	if (y1 >= (roi_top - MARGIN)) and (y2 <= (roi_bottom + MARGIN)):
		return True

	# Fallback: ratio of bbox height that overlaps the ROI (handles partial but mostly-inside bboxes)
	bbox_h = max(1, y2 - y1)
	overlap_h = max(0, min(y2, roi_bottom) - max(y1, roi_top))
	inside_ratio = float(overlap_h) / float(bbox_h)
	if inside_ratio >= RATIO_THRESH:
		return True

	return False


def _bbox_partially_inside(bbox, roi):
	x1, y1, x2, y2 = bbox
	return not (x2 < roi.get('x1', 0) or x1 > roi.get('x2', 0) or y2 < roi.get('y1', 0) or y1 > roi.get('y2', 0))


def observe_roi(vs, frame):
	"""Observer: update ROI/OCR-related state for a vehicle.

	Implements a per-track ROI state machine:
	  OUTSIDE -> ENTERING -> INSIDE -> EXITING -> DONE

	Requirements:
	- OCR only triggers when bbox is fully inside ROI and direction-specific
	  spatial conditions are satisfied and sustained for `MIN_STABLE_FRAMES`.
	- Observer must not modify counting-related fields (`has_entered`, `has_exited`,
	  `entry_count`, `exit_count`) or control flow.
	- Edge jitter allowed: small transient misses do not reset stability immediately.
	"""
	try:
		# Safety assertions required by the design
		assert vs.has_entered in (True, False)
		assert vs.has_exited in (True, False)

		bbox = getattr(vs, 'bbox', None)
		direction = getattr(vs, 'direction', None)
		if bbox is None or direction is None:
			return

		roi = getattr(config, 'PLATE_ROI', None)
		if roi is None:
			return

		# initialize FSM fields on vs
		if not hasattr(vs, 'roi_fsm_state'):
			vs.roi_fsm_state = 'OUTSIDE'
			vs.roi_fsm_inside = 0
			vs.roi_fsm_missed = 0
			# direction lock starts unset; will be set once and never recomputed
			vs.roi_direction_locked = None
			# keep OCR flags as-is if present
			vs.ocr_armed = getattr(vs, 'ocr_armed', False)
			vs.ocr_fired = getattr(vs, 'ocr_fired', False)

		# lock direction once and never recompute
		if getattr(vs, 'roi_direction_locked', None) is None:
			vs.roi_direction_locked = direction

		# smooth config values
		N = getattr(config, 'MIN_STABLE_FRAMES', 8)
		K = getattr(config, 'ROI_MISSED_TOLERANCE', 3)

		fully_inside = _bbox_fully_inside(bbox, roi)
		partially_inside = _bbox_partially_inside(bbox, roi)

		state = vs.roi_fsm_state

		# OUTSIDE -> ENTERING
		if state == 'OUTSIDE':
			if partially_inside:
				vs.roi_fsm_state = 'ENTERING'
				vs.roi_fsm_inside = 1 if fully_inside else 0
				vs.roi_fsm_missed = 0
				# NOTE: Do not buffer image frames here; main.py handles pixel operations.

		# ENTERING -> INSIDE or back to OUTSIDE
		elif state == 'ENTERING':
			if fully_inside:
				vs.roi_fsm_inside += 1
				vs.roi_fsm_missed = 0
				# main will capture pixels when OCR fires
				if vs.roi_fsm_inside >= N:
					vs.roi_fsm_state = 'INSIDE'
			elif partially_inside:
				# tolerate jitter; do not rapidly decrement inside counter
				vs.roi_fsm_missed = max(0, vs.roi_fsm_missed - 1)
				# do not buffer frames here
			else:
				vs.roi_fsm_missed += 1
				if vs.roi_fsm_missed > K:
					vs.roi_fsm_state = 'OUTSIDE'
					# Do not reset inside counter to zero on loss; decrement to add hysteresis
					vs.roi_fsm_inside = max(vs.roi_fsm_inside - 1, 0)
					vs.roi_fsm_missed = 0

		# INSIDE: require full containment for N consecutive frames; allow K missed
		elif state == 'INSIDE':
			if fully_inside:
				vs.roi_fsm_inside += 1
				vs.roi_fsm_missed = 0
				# do not buffer frames here
			elif partially_inside:
				# transient partial overlap
				vs.roi_fsm_missed = max(0, vs.roi_fsm_missed - 1)
				# do not buffer frames here
			else:
				vs.roi_fsm_missed += 1
				if vs.roi_fsm_missed > K:
					vs.roi_fsm_state = 'EXITING'
					vs.roi_fsm_missed = 0

			# OCR triggering isolated: only in INSIDE and only when full containment
			# for N frames and if OCR not yet fired. Do not change FSM state here.
			if (not vs.ocr_fired) and fully_inside and vs.roi_fsm_inside >= N:
				# spatial completeness enforcement: exact vertical containment
				x1, y1, x2, y2 = bbox
				if y1 >= roi.get('y1', -9999) and y2 <= roi.get('y2', 99999):
					# lock direction if not already locked
					if not getattr(vs, 'roi_direction_locked', None):
						vs.roi_direction_locked = direction
					# arm OCR
					vs.ocr_armed = True
					# Mark OCR as fired in FSM only. main.py is responsible for
					# performing the crop and creating the job.
					vs.ocr_fired = True

		# EXITING -> DONE or back to INSIDE (with hysteresis K)
		elif state == 'EXITING':
			if partially_inside:
				if fully_inside:
					vs.roi_fsm_state = 'INSIDE'
					vs.roi_fsm_inside = 1
					vs.roi_fsm_missed = 0
				else:
					# tolerate partial overlap jitter
					vs.roi_fsm_missed = max(0, vs.roi_fsm_missed - 1)
					# do not buffer frames here
			else:
				# not overlapping at all; increment missed and only go DONE after K
				vs.roi_fsm_missed += 1
				if vs.roi_fsm_missed > K:
					vs.roi_fsm_state = 'DONE'
					vs.roi_fsm_inside = 0
					vs.roi_fsm_missed = 0

		# DONE: keep buffer but do not re-trigger
		elif state == 'DONE':
			pass

	except Exception:
		# Do not let observer exceptions affect counting loop
		traceback.print_exc()
		return


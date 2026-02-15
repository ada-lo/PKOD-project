import cv2 as cv

def buffer_frame(vs, frame):
    """Append a copy of `frame` into `vs.ocr_frame_buffer`.

    Best-effort; never raise to avoid impacting live loop.
    """
    try:
        # store a copy to avoid retaining large referenced memory
        vs.ocr_frame_buffer.append(frame.copy())
    except Exception:
        # swallow errors to keep live loop robust
        pass

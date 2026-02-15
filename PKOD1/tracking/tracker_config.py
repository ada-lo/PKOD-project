import config

def create_tracker_config():
    """Create custom tracker configuration file."""
    if config.TRACKER_TYPE == "bytetrack":
        config_content = f"""tracker_type: bytetrack
track_high_thresh: {config.TRACK_HIGH_THRESH}
track_low_thresh: {config.TRACK_LOW_THRESH}
new_track_thresh: {config.NEW_TRACK_THRESH}
track_buffer: {config.TRACK_BUFFER}
match_thresh: {config.MATCH_THRESH}
fuse_score: True
"""
        config_file = "bytetrack_custom.yaml"
    else:  # botsort
        config_content = f"""tracker_type: botsort
track_high_thresh: {config.TRACK_HIGH_THRESH}
track_low_thresh: {config.TRACK_LOW_THRESH}
new_track_thresh: {config.NEW_TRACK_THRESH}
track_buffer: {config.TRACK_BUFFER}
match_thresh: {config.MATCH_THRESH}
cmc_method: sparseOptFlow
fuse_score: True
"""
        config_file = "botsort_custom.yaml"
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return config_file
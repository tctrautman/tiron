"""Public serving defaults."""

# Public checkpoint loaded by default.
MODEL_ID = "Trelis/tiron"
# Speaker encoder used to link local identities across windows.
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
# Sample rate expected by the acoustic model and speaker encoder.
SR = 16000
# Maximum input duration for one model decode.
CHUNK_MAX_SEC = 30.0
# Leading silence prevents deferred output on abrupt mid-word file onsets.
PAD_START_SEC = 0.75
# Token vocabulary exposes eight local speaker slots.
MAX_LOCAL_SPEAKERS = 8
# Linking is capped to the public model's eight speaker identities.
MAX_GLOBAL_SPEAKERS = 8
# Limit requests to three hours to bound memory and runtime.
MAX_AUDIO_SECONDS = 10800
# Merge small clusters only when their acoustic match is unambiguous.
DEFAULT_LOW_MASS_ECAPA_MERGE = True
# Prevent weak evidence from creating durable global identities.
DEFAULT_DEMOTE_WEAK_SPINE = True
# A surviving identity needs this much clean speech evidence.
DEMOTE_SPINE_MIN_MASS_SEC = 5.0
# A surviving identity should appear in more than one window.
DEMOTE_SPINE_MIN_WINDOWS = 2
# Staggered decoding calibrates speaker similarity per meeting by default.
TWO_PASS_DEFAULT = True
# The shorter first second-pass window offsets its boundaries from pass one.
TWO_PASS_B_FIRST_WINDOW_SEC = 15.0
# Unequal second-pass windows keep the two chunk grids from aligning.
TWO_PASS_B_WINDOW_SEC = 25.0
# Require enough same-speaker witnesses for stable calibration.
TWO_PASS_MIN_SAME = 30
# Demote calibrated clusters without enough accumulated speech.
TWO_PASS_DEMOTE_MASS_SEC = 10.0
# Demote calibrated clusters observed in too few windows.
TWO_PASS_DEMOTE_WINDOWS = 2

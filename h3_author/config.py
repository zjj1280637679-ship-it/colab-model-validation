# @title Author H3 kit: explicit, single-run settings (defaults are OFF)
MODE = "CPU_NO_WEIGHTS" # @param ["CPU_NO_WEIGHTS", "GPU_FULL_MODEL"]
ALLOW_THIS_RUN = False # @param {type:"boolean"}
GPU_BUDGET_MINUTES = 50 # @param {type:"integer"}
# Hard maximum 50 minutes total from setup start. CPU stays 12 minutes.
# 50 min at 8.88 CU/h is about 7.40 CU, not a price guarantee.
I_READ_H3_LICENSE = False # @param {type:"boolean"}
KNOWN_APPLICABLE_BLOCK_STATUS = "not_assessed" # @param ["not_assessed", "none_identified", "block_identified"]
HOST_REGION_STATUS = "unknown" # @param ["unknown", "reviewed"]
I_UNDERSTAND_UNRESOLVED_LICENSE_RISK = False # @param {type:"boolean"}
# none_identified means no clearly applicable blocking fact was identified for
# this deployment, not that the license has no territorial limits or grants are verified.
# The H3 license excludes EU/UK/Republic of Korea/USA from Applicable Territory.
# Unknown hosting region remains unknown; this notebook makes no legal conclusion.
# https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

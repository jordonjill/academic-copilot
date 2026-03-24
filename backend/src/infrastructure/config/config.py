# ===== 原有常量 =====
MAX_SEARCHES = 5
MAX_VALIDATION_ATTEMPTS = 3
MAX_TAVILY_SEARCHES = 10

# ===== STM 短期记忆常量 =====
STM_TOKEN_THRESHOLD = 6000   # 触发压缩的 token 阈值
STM_KEEP_RECENT = 6          # 压缩后保留的最近原文条数

# ===== 持久化路径 =====
DATA_DIR = "data"
USERS_DIR = "data/users"
CONVERSATION_DB = "data/conversations.db"

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

# 确保能找到 detail 目录 (以防万一)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 从 detail 包导入主逻辑
from detail import run_pipeline_logic

if __name__ == "__main__":
    try:
        run_pipeline_logic()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Fatal Error: {e}")
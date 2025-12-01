import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'master_3'))
sys.path.append(os.path.join(os.getcwd(), 'master_props'))
sys.path.append(os.path.join(os.getcwd(), 'v2'))

from master_3.api_utils import Master3Predictor

print("Initializing Master3Predictor...")
try:
    m3 = Master3Predictor()
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

import os
import sys

# Make `import src.*` work when running pytest from v3/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

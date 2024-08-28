import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, "/home/maximevaultier/stage/rag_python/app.py")

from app import app as application
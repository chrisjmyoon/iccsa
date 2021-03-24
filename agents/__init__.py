import pdb
import os

path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    __import__('.'.join([__name__, py]), fromlist=[py])
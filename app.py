import sys

# Reconfigure stdout encoding to UTF-8 to prevent UnicodeEncodeError in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("Welcome to Velox Vision 🏎️")

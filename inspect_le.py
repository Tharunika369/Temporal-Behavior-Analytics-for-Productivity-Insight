import joblib
from pathlib import Path

path = Path(r'c:\Users\tharu\Downloads\label_encoders.pkl')
if path.exists():
    enc = joblib.load(path)
    print('loaded', type(enc), 'keys', list(enc.keys()))
    for k,v in enc.items():
        print('===', k, '===')
        print('classes', getattr(v, 'classes_', None))
else:
    print('file not found')

from ultralytics import YOLO, checks, hub

if __name__ == '__main__':
    checks()

    hub.login('4f2936bc09a878fdea6113b64a0eef7fa01cd7781c')

    model = YOLO('https://hub.ultralytics.com/models/Jiv49CEvcGyiAbQsZlVI')
    results = model.train()
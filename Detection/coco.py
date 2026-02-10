import os
import urllib.request
import zipfile

# Папка для COCO
coco_dir = r"C:\Users\Utilisateur\Downloads\coco"
os.makedirs(coco_dir, exist_ok=True)

# Ссылки на архивы
files = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# Скачивание
for name, url in files.items():
    path = os.path.join(coco_dir, name)
    if not os.path.exists(path):
        print(f"Скачиваю {name}...")
        urllib.request.urlretrieve(url, path)
        print("Готово!")
    else:
        print(f"{name} уже скачан")

# Распаковка
for name in files.keys():
    path = os.path.join(coco_dir, name)
    print(f"Распаковываю {name}...")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(coco_dir)
    print("Готово!")

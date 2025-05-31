import os
import shutil

class Build_Folder:
    def __init__(self, input_folder, number_train, name_folder, name_file):
        '''
        input_folder: địa chỉ folder đầu vào
        number_train: số lượng ảnh train
        name_folder: tên folder con (ví dụ: cat, dog)
        name_file: tên file (ví dụ: cat, dog)
        '''
        self.input_folder = input_folder
        self.number_train = number_train
        self.name_folder = name_folder
        self.name = name_file

    def Design_folder(self):
        new_folder = "data"
        os.makedirs(new_folder, exist_ok=True)
        new_train = os.path.join(new_folder, "train")
        new_test = os.path.join(new_folder, "test")
        os.makedirs(new_train, exist_ok=True)
        os.makedirs(new_test, exist_ok=True)
        return new_train, new_test

    def build(self):
        new_train, new_test = self.Design_folder()
        files = os.listdir(self.input_folder)
        cnt = 0

        for file in files:
            src_path = os.path.join(self.input_folder, file)
            if os.path.isfile(src_path):  # Chỉ copy nếu là file

                if cnt < self.number_train:
                    folder_name = os.path.join(new_train, self.name_folder)
                else:
                    folder_name = os.path.join(new_test, self.name_folder)

                os.makedirs(folder_name, exist_ok=True)

                file_name = os.path.join(folder_name, f'{self.name}{cnt}.jpg')
                shutil.copy(src_path, file_name)
                cnt += 1

        print(f"Đã copy {cnt} ảnh vào data/train và data/test!")


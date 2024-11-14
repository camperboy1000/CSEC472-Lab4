import shutil
import os

train_directory = r"train"
test_directory = r"test"


def compile_train_directory(initial_directory):
    # f0001-f1499 & s0001-s1499

    for directory in os.listdir(initial_directory):
        directory = os.path.join(initial_directory, directory)
        for filename in os.listdir(directory):
            file_text = filename.split("_")[0]
            if file_text != "Thumbs.db":
                number = int(file_text[1:])

                file_src_path = os.path.join(directory, filename)
                file_dst_path = os.path.join(train_directory, filename)
                if number < 1500:
                    shutil.copyfile(file_src_path, file_dst_path)


def compile_test_directory(initial_directory):
    # f1501-f2000 & s1501-s2000

    for directory in os.listdir(initial_directory):
        directory = os.path.join(initial_directory, directory)
        for filename in os.listdir(directory):
            file_text = filename.split("_")[0]
            if file_text != "Thumbs.db":
                number = int(file_text[1:])

                file_src_path = os.path.join(directory, filename)
                file_dst_path = os.path.join(test_directory, filename)
                if number >= 1500:
                    shutil.copyfile(file_src_path, file_dst_path)


def main():
    # Needs changed to a relative path k thx
    initial_dir = r"C:\Users\username\Downloads\sd04\png_txt"

    compile_train_directory(initial_dir)
    compile_test_directory(initial_dir)


if __name__ == "__main__":
    main()

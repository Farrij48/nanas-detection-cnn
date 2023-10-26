import os

# Tentukan path folder
folder_path = "dataset5/nanas_mentah/"
# Dapatkan daftar semua file dalam folder
file_list = os.listdir(folder_path)

# Urutkan daftar file
sorted_files = sorted(file_list)

# Tentukan nomor awal urutan
nomor_urutan = 1

# Loop melalui setiap file dalam folder
for file_name in sorted_files:
    # Dapatkan ekstensi file (jika ada)
    file_name_parts = os.path.splitext(file_name)
    extension = file_name_parts[1]

    # Bentuk nama file baru dengan format "file{nomor_urutan}{ekstensi}"
    new_name = f"nanas_mentah{nomor_urutan}{extension}"

    # Dapatkan path lengkap dari file lama dan file baru
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)

    # Ubah nama file
    os.rename(old_path, new_path)
    print(f"File '{file_name}' telah diubah menjadi '{new_name}'")

    # Tingkatkan nomor urutan untuk file berikutnya
    nomor_urutan += 1


# # import os

# # # Tentukan path utama folder
# # main_folder_path = "dataset3/fruits-360/Training/"

# # # Dapatkan daftar folder di dalam folder utama
# # folders = [
# #     folder
# #     for folder in os.listdir(main_folder_path)
# #     if os.path.isdir(os.path.join(main_folder_path, folder))
# # ]

# # # Loop melalui setiap folder
# # for folder_name in folders:
# #     # Tentukan path folder saat ini
# #     folder_path = os.path.join(main_folder_path, folder_name)

# #     # Dapatkan daftar semua file dalam folder
# #     file_list = os.listdir(folder_path)

# #     # Urutkan daftar file
# #     sorted_files = sorted(file_list)

# #     # Tentukan nomor awal urutan
# #     nomor_urutan = 1

# #     # Loop melalui setiap file dalam folder
# #     for file_name in sorted_files:
# #         # Dapatkan ekstensi file (jika ada)
# #         file_name_parts = os.path.splitext(file_name)
# #         extension = file_name_parts[1]

# #         # Bentuk nama file baru dengan format "file{nomor_urutan}{ekstensi}"
# #         new_name = f"{folder_name}_{nomor_urutan}{extension}"

# #         # Dapatkan path lengkap dari file lama dan file baru
# #         old_path = os.path.join(folder_path, file_name)
# #         new_path = os.path.join(folder_path, new_name)

# #         # Ubah nama file
# #         os.rename(old_path, new_path)
# #         print(
# #             f"File '{file_name}' dalam folder '{folder_name}' telah diubah menjadi '{new_name}'"
# #         )

# #         # Tingkatkan nomor urutan untuk file berikutnya
# #         nomor_urutan += 1

# import os

# # Tentukan path utama folder
# main_folder_path = "dataset3/fruits-360/Training/"

# # Dapatkan daftar folder di dalam folder utama
# folders = [
#     folder
#     for folder in os.listdir(main_folder_path)
#     if os.path.isdir(os.path.join(main_folder_path, folder))
# ]

# # Loop melalui setiap folder
# for folder_name in folders:
#     # Ubah spasi dalam nama folder menjadi underscore
#     folder_name_with_underscore = folder_name.replace(" ", "_")

#     # Tentukan path folder saat ini
#     folder_path = os.path.join(main_folder_path, folder_name)

#     # Dapatkan daftar semua file dalam folder
#     file_list = os.listdir(folder_path)

#     # Urutkan daftar file
#     sorted_files = sorted(file_list)

#     # Tentukan nomor awal urutan
#     nomor_urutan = 1

#     # Loop melalui setiap file dalam folder
#     for file_name in sorted_files:
#         # Dapatkan ekstensi file (jika ada)
#         file_name_parts = os.path.splitext(file_name)
#         extension = file_name_parts[1]

#         # Bentuk nama file baru dengan format "folder_nama_file{nomor_urutan}{ekstensi}"
#         new_name = f"{folder_name_with_underscore}_{nomor_urutan}{extension}"

#         # Dapatkan path lengkap dari file lama dan file baru
#         old_path = os.path.join(folder_path, file_name)
#         new_path = os.path.join(folder_path, new_name)

#         # Ubah nama file
#         os.rename(old_path, new_path)
#         print(
#             f"File '{file_name}' dalam folder '{folder_name}' telah diubah menjadi '{new_name}'"
#         )

#         # Tingkatkan nomor urutan untuk file berikutnya
#         nomor_urutan += 1

# import os
# import shutil

# # Tentukan path utama folder
# main_folder_path = "dataset3/fruits-360/Training/"

# # Dapatkan daftar folder di dalam folder utama
# folders = [
#     folder
#     for folder in os.listdir(main_folder_path)
#     if os.path.isdir(os.path.join(main_folder_path, folder))
# ]

# # Buat folder khusus untuk menyimpan file yang diubah
# output_folder_path = "dataset3/fruits-360/Modified/"

# # Cek apakah folder output sudah ada, jika belum, buat folder tersebut
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

# # Loop melalui setiap folder
# for folder_name in folders:
#     # Ubah spasi dalam nama folder menjadi underscore
#     folder_name_with_underscore = folder_name.replace(" ", "_")

#     # Tentukan path folder saat ini
#     folder_path = os.path.join(main_folder_path, folder_name)

#     # Dapatkan daftar semua file dalam folder
#     file_list = os.listdir(folder_path)

#     # Urutkan daftar file
#     sorted_files = sorted(file_list)

#     # Tentukan nomor awal urutan
#     nomor_urutan = 1

#     # Loop melalui setiap file dalam folder
#     for file_name in sorted_files:
#         # Dapatkan ekstensi file (jika ada)
#         file_name_parts = os.path.splitext(file_name)
#         extension = file_name_parts[1]

#         # Bentuk nama file baru dengan format "folder_nama_file{nomor_urutan}{ekstensi}"
#         new_name = f"{folder_name_with_underscore}_{nomor_urutan}{extension}"

#         # Dapatkan path lengkap dari file lama dan file baru
#         old_path = os.path.join(folder_path, file_name)
#         new_path = os.path.join(output_folder_path, new_name)

#         # Ubah nama file dan pindahkan ke folder khusus
#         os.rename(old_path, new_path)
#         print(
#             f"File '{file_name}' dalam folder '{folder_name}' telah diubah menjadi '{new_name}' dan dipindahkan ke folder khusus."
#         )

#         # Tingkatkan nomor urutan untuk file berikutnya
#         nomor_urutan += 1

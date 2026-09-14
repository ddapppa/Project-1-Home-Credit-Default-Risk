#  Menjadikan modeling/ sebagai Package
# Tujuan kode: File ini wajib ada (walau isinya minim) agar Python mengenali folder modeling/ 
# sebagai package yang bisa di-import. Saya isi kosong dulu — nanti setelah beberapa fungsi 
# trainer per model selesai, baru kita tambahkan baris from .train import 
# train_logistic_regression dsb di sini supaya notebook bisa from modeling 
# import train_logistic_regression (lebih rapi daripada from modeling.train 
# import train_logistic_regression).

"""
src/modeling/__init__.py
Menjadikan folder `modeling/` sebagai Python package.
Export fungsi trainer akan ditambahkan di sini setelah setiap
trainer model selesai dibuat dan direview, contoh nanti:

    from .train import train_logistic_regression

Sengaja dikosongkan dulu — jangan tambahkan logika training di file ini.
"""
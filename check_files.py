import os

folder = "final_ach/captured"
print("📂 Listing files in:", os.path.abspath(folder))
print(os.listdir(folder))

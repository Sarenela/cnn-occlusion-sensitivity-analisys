from pathlib import Path

cats_path = Path("./cats")
dogs_path = Path("./dogs")

num_cats = len(list(cats_path.glob("*.jpg")))
num_dogs = len(list(dogs_path.glob("*.jpg")))

print(f"Number of cat images: {num_cats}")
print(f"Number of dog images: {num_dogs}")

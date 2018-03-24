# save a dictionary into a pickle file
import pickle
favorite_color = {"lion": "yellow", "kitty":"red"}
# Write a pickled representation of obj to the open file object file
# “wb” to write it, and “rb” to read it.
pickle.dump(favorite_color, open("save.p", "wb"))

# load dictionary back form the pickle file
color_load = pickle.load(open("save.p", "rb"))
print(color_load)
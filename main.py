# main.py
# Simple terminal menu for LeafGuard AI.

from train import train
from predict import predict


def show_menu():
    print("=================================")
    print("        LeafGuard AI 🌿")
    print("=================================")
    print("1. Train the model")
    print("2. Predict a leaf disease")
    print("3. Exit")


def main():
    while True:
        show_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            train()
        elif choice == "2":
            path = input("Enter the path of the leaf image: ").strip()
            predict(path)
        elif choice == "3":
            print("Goodbye! 🌿")
            break
        else:
            print("Please choose 1, 2 or 3.")
        print()  # empty line before the menu shows again


# Start the program.
if __name__ == "__main__":
    main()

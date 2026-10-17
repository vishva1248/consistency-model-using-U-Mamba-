mode  = input("Enter the the mode Training / Testing :- ")
while True:
    if mode.lower() == "training":
        from Training import Training
        break
    elif mode.lower() == "testing":
        print("testing")
        from testing_pure_noise import Testing_pure_noise
        Testing_pure_noise()
        break
    else:
        print("Invalid mode entered")
        continue
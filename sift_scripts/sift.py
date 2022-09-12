import cv2
import functions
import img_proc

if __name__ == '__main__':

    # Get data path
    # data_path = input("Give file path to images : ")
    # data_path.strip("\n")
    data_path = "training_images"

    # Load images and save in lists
    init_images, label_list = functions.load_images(data_path)

    # Choose one of the photos to experiment with
    while True:

        chosen_photo = input("Which photo would you like to choose for the experiment? ")
        # chosen_photo.strip("\n")
        if chosen_photo == "dog":
            chosen_label = 'd'
            break
        elif chosen_photo == "cat":
            chosen_label = 'k'
            break
        elif chosen_photo == "wheel":
            chosen_label = 'w'
            break
        elif chosen_photo == "tree":
            chosen_label = 't'
            break
        else:
            print("This photo doesn't exist. Try again.")

    # Perform transformations on the chosen photo
    training_list = img_proc.process_img(init_images, label_list, chosen_label)

    flag = 1
    num_feat_list = []
    num_octave_list = []
    contrast_list = []
    edge_list = []
    gauss_sig_list = []
    init_keypoints = []
    success = []
    selection_list = []
    loops = 1

    # If you choose not to experiment, a SIFT model with default parameters will be created
    # If you choose to experiment, it would be best to experiment
    # with one parameter only while keeping the others steady
    answer = input("Do you want to experiment with parameters or not? (yes or no) ")

    if answer == "no":

        # SIFT model with default parameters
        print("\nBuilding a model...")
        sift_model = cv2.SIFT_create()
        _, _ = functions.apply_sift_algo(sift_model, training_list)

    elif answer == "yes":

        while flag:

            num_feat = int(input("\nGive number of features: "))
            num_feat_list.append(num_feat)

            num_octave = int(input("Give number of octave layers: "))
            num_octave_list.append(num_octave)

            contrast = float(input("Give contrast threshold: "))
            contrast_list.append(contrast)

            edge = float(input("Give edge threshold: "))
            edge_list.append(edge)

            gauss_sig = float(input("Give Gaussian sigma: "))
            gauss_sig_list.append(gauss_sig)

            # SIFT model
            print("\nBuilding a model...")
            sift_model = cv2.SIFT_create(
                nfeatures=num_feat, nOctaveLayers=num_octave,
                contrastThreshold=contrast, edgeThreshold=edge, sigma=gauss_sig
            )

            print("\nLoop ", loops)
            keyp, rate = functions.apply_sift_algo(sift_model, training_list)
            init_keypoints.append(keyp)
            success.append(rate)

            # If yes, you create a new model, with new parameter
            repetition = input("\nWould you like to repeat the experiment? (yes or no) ")
            if repetition == "yes":
                loops += 1
                continue

            # If no, you can have diagrams drawn or not and then the program ends
            elif repetition == "no":
                diagram = input("Would you like to have diagrams of your experiments? (yes or no) ")

                # Diagrams will be saved in the same file the code file is
                if diagram == "yes":

                    # 1. nfeatures, 2. nOctaveLayers, 3. contrastThreshold, 4. edgeThreshold or 5. sigma
                    selection = int(input("Give the parameter you have been experimenting with: "))
                    if selection == 1:
                        selection_list = num_feat_list.copy()
                        param = "Number of Features"
                    elif selection == 2:
                        selection_list = num_octave_list.copy()
                        param = "Number of Octave Layers"
                    elif selection == 3:
                        selection_list = contrast_list.copy()
                        param = "Contrast Threshold"
                    elif selection == 4:
                        selection_list = edge_list.copy()
                        param = "Edge Threshold"
                    elif selection == 5:
                        selection_list = gauss_sig_list.copy()
                        param = "Gaussian Sigma"
                    else:
                        print("This answer is not acceptable. Exiting...")
                        exit(0)

                    functions.save_graphs(selection_list, success, init_keypoints, param)

                    break

                flag = 0

    else:
        print("Not acceptable answer. Exiting...")

    print("\nFinished! Exiting...")

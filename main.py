import cv2
import numpy as np
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import convolve2d
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import joblib

def get_list_videos():
    # Get video list in training and test sets
    train_data_directory = r'D:\StairCase_Estimation\Train_Test_Validation_Split\folder_training'
    test_data_directory = r'D:\StairCase_Estimation\Train_Test_Validation_Split\folder_test'
    train_video_list = os.listdir(train_data_directory)
    test_video_list = os.listdir(test_data_directory)
    return train_video_list+test_video_list


def compute_and_estimate_steps(video_file_name, distance_list, validation =False):
    # Check if y-distance is below the threshold
    # prev_distance = length_between_ankles
    # return prev_distance, step_counter, threshold
    # Should fine tune parameters
    peaks = find_peaks(distance_list, prominence=0.02, distance=8)
    print("Peaks position:", peaks[0])
    print('Peak count', len(peaks))

    # Plotting
    plt.plot(distance_list)
    plt.title("Finding Peaks")

    [plt.axvline(p, c='C3', linewidth=0.3) for p in peaks[0]]

    if validation:
        plt.savefig(r'D:\StairCase_Estimation\Model_and_Results\Validation_analysis_data' + video_file_name + '_peaks.png')
    else:
        plt.savefig(r'D:\StairCase_Estimation\Data_Extraction\Pose_estimation_peaks_Analysis\\' + video_file_name + '_peaks.png')
    plt.clf()
    return len(peaks[0])

def compute_speed(step_counter, duration):
    #Assuming height of the stairs in meters
    # Average stair height is 7.5 and stair depth is 9 inches - This gives stair length as 11.5 inch or approx 30 cm
    stair_length = 0.3
    total_length_covered = stair_length * step_counter
    # Speed in m/s
    speed = float(total_length_covered/duration)
    return speed, float(step_counter/duration)


# def save_pe_data(video_path, timestamp, length_right_hip_to_knee, length_left_hip_to_knee):

def data_extraction(video_list):
    # Open the file in write mode to create an empty file
    new_dict = {'File_name':[],'Magnitude':[],'Angle':[],'Shape':[],'Left_foot':[],'Right_foot':[],'Length_between_ankles':[], 'Convolution_result':[]}
    # new_df.to_csv(r'D:\StairCase_Estimation\DenseFlow_Tensors\denseflow_pose_data.csv', mode='w', header=True, index=False)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    for idx,video_path in enumerate(video_list):
        # Get a VideoCapture object from video and store it in vs
        file_name_list = []
        magnitude_list = []
        angle_list = []
        shape_list = []
        timestamp_list = []
        length_between_ankles_list = []
        right_foot_list = []
        left_foot_list = []
        frame_count = 0
        video_path = r'D:\StairCase_Estimation\Dataset' + r'\\' + video_path

        vc = cv2.VideoCapture(video_path)
        fps = vc.get(cv2.CAP_PROP_FPS)
        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        video_time_elapsed = total_frames/fps
        # Read first frame
        ret, first_frame = vc.read()
        if not ret:
            continue
        # Scale and resize image
        frame_count += 1

        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Create mask
        mask = np.zeros_like(first_frame)
        # Sets image saturation to maximum
        mask[..., 1] = 255

        # out = cv2.VideoWriter(output_video_path, -1, 1, (600, 600))
        print('Starting dense flow computation for file ',video_path)
        while (vc.isOpened()):
            # Read a frame from video
            ret, frame = vc.read()
            if not ret:
                break
            # Convert the frame to RGB (MediaPipe requires RGB input)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            if results.pose_landmarks is not None:
                frame_count += 1
                # Extract landmark coordinates
                landmarks = results.pose_landmarks.landmark

                # Get the coordinates of specific landmarks
                right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                # Calculate the lengths of different limbs
                length_between_ankles = ((right_foot.x - left_foot.x) ** 2 + (right_foot.y - left_foot.y) ** 2) ** 0.5
                length_between_ankles_list.append(length_between_ankles)
                right_foot_list.append([right_foot.x, right_foot.y])
                left_foot_list.append([left_foot.x, left_foot.y])
                timestamp = vc.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_list.append(timestamp)
                # Convert new frame format`s to gray scale and resize gray frame obtained
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = cv2.resize(gray, None, fx=scale, fy=scale)

                # Calculate dense optical flow by Farneback method
                # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,  None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Compute the magnitude and angle of the 2D vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                file_name = os.path.basename(video_path)
                file_name_list.append(file_name)
                magnitude_list.append(magnitude.flatten())
                angle_list.append(angle.flatten())
                shape_list.append(np.shape(magnitude.flatten())[0])
                new_dict['File_name'].append(os.path.basename(video_path))
                new_dict['Magnitude'].append(magnitude.flatten())
                new_dict['Angle'].append(angle.flatten())
                new_dict['Shape'].append(np.shape(magnitude.flatten())[0])
                new_dict['Left_foot'].append([left_foot.x, left_foot.y])
                new_dict['Right_foot'].append([right_foot.x, right_foot.y])
                new_dict['Length_between_ankles'].append(length_between_ankles)
                convolution_result = compute_convolution(magnitude, angle)
                new_dict['Convolution_result'].append(convolution_result[0][0])

                # Set image hue according to the optical flow direction
                mask[..., 0] = angle * 180 / np.pi / 2
                # Set image value according to the optical flow magnitude (normalized)
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                # Convert HSV to RGB (BGR) color representation

                prev_gray = gray
                # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        # The following frees up resources and closes all windows
        vc.release()
        cv2.destroyAllWindows()
        # Calculating steps count from peak signals
        video_file_name = os.path.basename(video_path)
        step_counter = compute_and_estimate_steps(video_file_name, length_between_ankles_list)

        # Calculating speed from step counter
        # print('Step Counter', step_counter)
        # print('Video time elapsed', video_time_elapsed)
        speed_in_meter_per_second, speed_in_steps_per_second = compute_speed(step_counter, video_time_elapsed)
        plt.plot(timestamp_list, length_between_ankles_list)
        plt.xlabel('Timestamp (milliseconds)')
        plt.ylabel('Distance (pixels)')
        plt.title('Distance between Left and Right Feet over Time')
        plt.savefig(r'D:\StairCase_Estimation\Data_Extraction\Pose_estimation_peaks_Analysis\\' + video_file_name + '_analysis.png')
        plt.clf()
        print('-------------  Percentage completed ---------------', round(float(idx / len(video_list) * 100), 2))
    denseflow_df = pd.DataFrame(new_dict)
    return denseflow_df

def compute_convolution(magnitude, angle):
    result = convolve2d(magnitude, angle, mode='valid')
    # print(result[0])
    # print(result[0][0])
    # print(result.shape)
    # print(type(result))
    return result

def data_preparation_for_training(dense_flow_df):
    dense_flow_df = dense_flow_df[['Convolution_result','Length_between_ankles']]
    dense_flow_df.to_csv(r'D:\StairCase_Estimation\Data_Extraction\machine_learning_data.csv', index=None)
    return dense_flow_df

def random_forest_regressor(dense_flow_df=None):
    # Load your dataset
    print('****************** Starting training ***************************')
    # Create a feature matrix X and target vector y
    if dense_flow_df is None:
        dense_flow_df = pd.read_csv(r'D:\StairCase_Estimation\Data_Extraction\machine_learning_data.csv')
    X = dense_flow_df['Convolution_result'].values.flatten()
    y = dense_flow_df['Length_between_ankles'].values.flatten()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pd.DataFrame({'Convolution_result':X_test,'Length_between_ankles':y_test}).to_csv(r"D:\StairCase_Estimation\Data_Extraction\test_data.csv")
    # Create a Random Forest regressor
    rf_regressor = RandomForestRegressor()

    # Define the hyperparameter grid for tuning
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(rf_regressor, param_grid, scoring='neg_mean_absolute_percentage_error', cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train.reshape(-1,1), y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create the final model with the best hyperparameters
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_train.reshape(-1,1), y_train)

    # Save model
    joblib.dump(final_model, r"D:\StairCase_Estimation\Model_and_Results\Model\random_forest_model.joblib")
    # Make predictions on the test set
    y_pred = final_model.predict(X_test.reshape(-1,1))

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Best Hyperparameters: {best_params}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    APE = []

    # Iterate over the list values
    for idx, t in enumerate(y_test):
        # Calculate percentage error
        per_err = (y_test[idx] - y_pred[idx]) / y_test[idx]

        # Take absolute value of
        # the percentage error (APE)
        per_err = abs(per_err)

        # Append it to the APE list
        APE.append(per_err)

        # Calculate the MAPE
    MAPE = sum(APE) / len(APE)

    # Print the MAPE value and percentage
    print(f''' 
    MAPE   : {round(MAPE, 2)} 
    MAPE % : {round(MAPE * 100, 2)} % 
    ''')
    # Specify the file path
    file_path = r"D:\StairCase_Estimation\Model_and_Results\model_results.txt"
    values_to_save = [f'Best parameters: {best_params}',f'Mean absolute error: {mae}', f'MAPE: {MAPE}',f'MAPE % : {round(MAPE * 100, 2)}']
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each value to the file
        for value in values_to_save:
            file.write(value + "\n")
    # Plot the results
    plt.scatter(X_test, y_test, label='Actual data')
    plt.scatter(X_test, y_pred, color='red', label='Random Forest Prediction')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.savefig(r'D:\StairCase_Estimation\Model_and_Results\prediction_analysis.png')

def calculate_MSE():
    model_path = r'D:\StairCase_Estimation\Model_and_Results\Model\random_forest_model.joblib'
    loaded_model = joblib.load(model_path)
    test_df = pd.read_csv(r'D:\StairCase_Estimation\Data_Extraction\test_data.csv')
    y_actual = test_df['Length_between_ankles']
    y_pred = loaded_model.predict(np.array(test_df['Convolution_result']).reshape(-1,1))
    MSE = np.square(np.subtract(y_actual, y_pred)).mean()
    R2_score = r2_score(y_actual, y_pred)
    print('*******MSE********', MSE)
    print('*******R2*********', R2_score)

def testing_pipeline():
    # Load the trained and saved model
    model_path = r'D:\StairCase_Estimation\Model_and_Results\Model\random_forest_model.joblib'
    loaded_model = joblib.load(model_path)

    # Take validation files
    validation_data_directory = r'D:\StairCase_Estimation\Train_Test_Validation_Split\folder_validation'
    validation_video_list = os.listdir(validation_data_directory)
    final_data_dict = {'File_name':[],'Steps_climbed':[],'Speed in step/second':[],'Speed in meter/second':[]}
    # For each video - compute magnitude and angle vector , convolution result, total duration
    for idx,video_path in enumerate(validation_video_list):
        # Get a VideoCapture object from video and store it in vs
        video_path = r'D:\StairCase_Estimation\Dataset\\' + video_path
        convolution_result_list = []
        vc = cv2.VideoCapture(video_path)
        fps = vc.get(cv2.CAP_PROP_FPS)
        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        video_time_elapsed = total_frames/fps
        # Read first frame
        ret, first_frame = vc.read()
        if not ret:
            continue
        # Convert to gray scale
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Create mask
        mask = np.zeros_like(first_frame)
        # Sets image saturation to maximum
        mask[..., 1] = 255

        # out = cv2.VideoWriter(output_video_path, -1, 1, (600, 600))
        print('Starting dense flow computation for file ',video_path)
        while (vc.isOpened()):
            # Read a frame from video
            ret, frame = vc.read()
            if not ret:
                break
            # Convert the frame to RGB (MediaPipe requires RGB input)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert new frame format`s to gray scale and resize gray frame obtained
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.resize(gray, None, fx=scale, fy=scale)
           # Calculate dense optical flow by Farneback method
            # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,  None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Compute the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            convolution_result = compute_convolution(magnitude, angle)
            convolution_result_list.append(convolution_result[0][0])

            # Set image hue according to the optical flow direction
            mask[..., 0] = angle * 180 / np.pi / 2
            # Set image value according to the optical flow magnitude (normalized)
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # Convert HSV to RGB (BGR) color representation

            prev_gray = gray
            # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # The following frees up resources and closes all windows
        vc.release()
        cv2.destroyAllWindows()
        # Create a validation df with convoluted result

        # Predict the distance between ankles
        y_pred = loaded_model.predict(np.array(convolution_result_list).reshape(-1, 1))
        # For each video - create a list of distance bw ankles
        step_counter = compute_and_estimate_steps(os.path.basename(video_path), y_pred, True)

        # Compute the steps and speed
        average_speed_mps, average_speed_steps_per_second = compute_speed(step_counter, video_time_elapsed)
        # Create a csv that has video name and total speed in m/s and steps/second
        final_data_dict['File_name'].append(video_path)
        final_data_dict['Steps_climbed'].append(step_counter)
        final_data_dict['Speed in meter/second'].append(average_speed_mps)
        final_data_dict['Speed in step/second'].append(average_speed_steps_per_second)
    pd.DataFrame(final_data_dict, index=None).to_csv(r'D:\StairCase_Estimation\Model_and_Results\validation_result.csv')
if __name__ == '__main__':
    dataset_acquired = False
    if dataset_acquired:
        random_forest_regressor()
    else:
        video_lists = get_list_videos()
        dense_flow_df = data_extraction(video_lists)
        entire_dataset = data_preparation_for_training(dense_flow_df)
        # Perfrom XGBoost
        random_forest_regressor(entire_dataset)

    testing_pipeline()
    calculate_MSE()
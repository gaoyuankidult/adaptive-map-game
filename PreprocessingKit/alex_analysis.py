import numpy as np
import pandas as pd
import cv2


if __name__ == "__main__":
    info_df = pd.read_csv('../data/2019_05_03/09-05-21/gazeData_world.tsv','\t')
    gaze_count = 0
    current_frame = 0 
    next_frame = 0
    largest_frame = 0
    
    vidcap = cv2.VideoCapture('../data/2019_05_03/09-05-21/worldCamera.mp4')
    video_count = 0

    image_count = 0

    data = {'gaze_x':[],'gaze_y':[]}

    while True:
        try:
            while True:
                current_frame = int(info_df.iloc[gaze_count]["frame_idx"])
                next_frame =  int(info_df.iloc[gaze_count+1]["frame_idx"])
                gaze_count += 1
                #print(next_frame,largest_frame)  
                #if next_frame > largest_frame:
                #    largest_frame = next_frame
                #    break
                if next_frame > current_frame and next_frame > largest_frame and current_frame >= 1:
                    largest_frame = next_frame
                    break
        except:
            print("Reached the end")
            break

        while True:
            print("Loop detector", video_count,current_frame,next_frame, gaze_count, largest_frame,video_count == current_frame)
            success,image = vidcap.read()
            video_count += 1
            if video_count == current_frame:
                break
        
        if int(info_df.iloc[gaze_count]["confidence"]) != 1:
            continue

        height, width = image.shape[:2]

        cv2.circle(image,(int(info_df.iloc[gaze_count]["norm_pos_x"]*width),
                        int(info_df.iloc[gaze_count]["norm_pos_y"]*height)), 30, (0,0,255), 2)  
        cv2.imwrite("../result_example/image%d.jpg"%image_count, image)

        data['gaze_x'].append(int(info_df.iloc[gaze_count]["norm_pos_x"]*width))
        data['gaze_y'].append(int(info_df.iloc[gaze_count]["norm_pos_y"]*height))
        cv2.imwrite("../original_example/image%d.jpg"%image_count, image)
        image_count += 1
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        
        print("Gaze count:", gaze_count)
        print("Video count:", video_count)
        print("Current frame:", current_frame)

    df = pd.DataFrame(data)
    df.to_csv('../gaze_labels.cvs')
    

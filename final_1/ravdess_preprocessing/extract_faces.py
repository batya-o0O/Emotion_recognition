import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=(720, 1280), device=device)

root = 'C:/Users/zhk27/OneDrive/Рабочий стол/Minho'
total_frames = 600  # Total frames to take from the start
parts = 6  # Number of parts to divide into
frames_per_part = total_frames // parts  # 100 frames per part
frames_to_select_per_part = 15  # Frames to select per part
step = frames_per_part // frames_to_select_per_part  # Step for frame selection

failed_videos = []

# Iterate over sessions in the root directory
for sess in tqdm(sorted(os.listdir(root))):   
    # Iterate over files in each session
    for filename in os.listdir(os.path.join(root, sess)):
        # Check if the file is an mp4 video and contains "Speaking" in its name
        if filename.endswith('.mp4') and "Speaking" in filename:
            video_path = os.path.join(os.path.join(root, sess), filename)
            cap = cv2.VideoCapture(video_path) 

            part_frame = 0  # Counter to keep track of the part number
            frame_counter = 0  # Overall frame counter
            all_frames_to_select=[]
            
            # Generate a list of frames to select from each part
            for i in range(0,600,100):
                for j in range(frames_to_select_per_part):
                    all_frames_to_select.append(i+j*7)
            
            # Iterate over the frames to select
            for cri in range(0,len(all_frames_to_select),15):
                if cri+15>len(all_frames_to_select):
                    break
                frames_to_select=all_frames_to_select[cri:cri+15]
                numpy_video = []
                frame_ctr = 0

                while True: 
                    ret, im = cap.read()
                    if not ret:
                        break
                    # Check if the current frame is in the frames to select list
                    if frame_ctr not in frames_to_select:
                        frame_ctr += 1
                        continue
                    else:
                        frames_to_select.remove(frame_ctr)
                        frame_ctr += 1

                    try:
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    except:
                        break
                    temp = im[:,:,-1]
                    im_rgb = im.copy()
                    im_rgb[:,:,-1] = im_rgb[:,:,0]
                    im_rgb[:,:,0] = temp
                    im_rgb = torch.tensor(im_rgb)
                    im_rgb = im_rgb.to(device)

                    bbox = mtcnn.detect(im_rgb)
                    if bbox[0] is not None:
                        bbox = bbox[0][0]
                        bbox = [round(x) for x in bbox]
                        x1, y1, x2, y2 = bbox
                        im = im[y1:y2, x1:x2, :]
                        im = cv2.resize(im, (224,224))
                        numpy_video.append(im)
                
                # If there are remaining frames to select, add blank frames to the numpy_video list
                if len(frames_to_select) > 0:
                    for i in range(len(frames_to_select)):
                        numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
                
                # Save the numpy_video array as a .npy file
                r=filename[:-4]
                nr=r.split('_')
                nr2='_'.join(nr[:2]+nr[3:])
                np.save(os.path.join(os.path.join(root,sess),nr2+"_"+f"{part_frame}"+'_facecroppad.npy'), np.array(numpy_video))
                
                # Check if the number of frames in numpy_video is equal to the expected number
                if len(numpy_video) != 15:
                    print('Error', os.path.join(root), filename)   
                
                part_frame+=1
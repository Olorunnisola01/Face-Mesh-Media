#Importing the necessary libraries
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
from PIL import Image
import kaggle



#Integrating streamlit and mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = 'demo.mp4'
OUTM = 'output.mp4'
DEMO_IMAGE = 'demo.jpg'


st.title('Face Mesh Application using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.title('Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    #Set the dimensions of the image to be resized and store the size of the image."
    dim = None
    (h, w) = image.shape[:2]

    #If neither the width nor the height is specified, return the original image unchanged
    if width is None and height is None:
        return image

    # Determine if the value for the width is equal to None
    if width is None:
        # Compute the ratio between the height and width, and use it to determine the dimensions of the object
        r = height / float(h)
        dim = (int(w * r), height)

    # If the height is not specified, it is set to None.
    else:
        # Calculate the widths' ratio and use it to determine the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized




app_mode = st.sidebar.selectbox('Choose the App mode',
['About the App','Run the App on a Video','Run the App on an Image']
)

if app_mode =='About the App':

    st.markdown('This application uses Google''s MediaPipe to generate a Face Mesh on a video')  

    st.text('A demonstration of the output for a face mesh')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
 
    st.markdown('''
          # About Author \n 
             As a mechanical engineer with a strong interest in intelligent systems, I am actively seeking opportunities to contribute to the field through research and collaboration. My passion for developing smart solutions to complex problems drives my pursuit of expertise in this area. Please feel free to contact me at Olorunnisola01@gmail.com to discuss potential job opportunities or collaborative projects.''')
elif app_mode =='Run the App on a Video':

    st.subheader('We are using Face Mesh on a video')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    
    st.sidebar.text('Params For video')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    #max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces',value =1,min_value= 1)

    st.markdown(' ## Output')
    stframe = st.empty()
    

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

    

    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)



    

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps, (width, height))


    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    

    with mp_face_mesh.FaceMesh(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence , 
    max_num_faces = max_faces) as face_mesh:




        while vid.isOpened():

            ret, frame = vid.read()

            if not ret:
                continue
                
            

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            

            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list=face_landmarks,
                    # connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)


            out.write(frame)    
            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)

            frame = image_resize(image = frame, width = 640)
            
            stframe.image(frame,channels = 'BGR',use_column_width=True)

    

    st.text('Video Processed')

    output_video = open('output1.webm','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)


        





    vid.release()
    out.release()


elif app_mode =='Run the App on an Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
    st.subheader('We are applying Face Mesh on an Image')

    st.sidebar.text('Params for Image')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    max_faces = st.sidebar.number_input('Maximum Number of Faces',value =2,min_value = 1)



    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)


    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:
    


        results = face_mesh.process(image)

        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            #print('face_landmarks:', face_landmarks)

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            # connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        st.subheader('Output Image')

        

        st.image(out_image,use_column_width= True)

        

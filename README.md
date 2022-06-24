# Robo-Romeo

![image](https://user-images.githubusercontent.com/103648207/175067813-01692494-a157-495d-878a-24c2fe23c356.png)

#  Project Description - can AI be creative?

Project exploring the creative abilities of AI: generating captions from images and turning the captions into romantic poetry.

## Solution structure

- Utilised CNN model (EfficientNetB0) to encode images into vectors and added embedding layer to tokenize the captions corresponding with images.
- Established the LSTM model and trained it using Google Cloud Platform (GCP) Vertex AI to predict the next word of sequences and output whole sentences.
- Applied Object-Oriented Programming (OOP) to design the batch for training dataset.
- Used +118k images and +500k captions to train model.
- Built up the scoring function which using doc2vec to transfer sentences into vectors and calculate the cosine similarities to evaluate the performance of image captioning. 
- Imported open API called GPT-3 to output the beautiful poetry according to information gathered from images. 
- Developed a website using Streamlit to present both poetry and robot voices.
- Used Text-to-speech API (Uberduck.io) to provide audio output for poetry


# Bonus - Attention layer

Custom coded layer added to the model to introduce attention mechanism in order to improve the descriptive accuracy of captioning.

<img width="716" alt="Screenshot 2022-06-21 at 17 34 46" src="https://user-images.githubusercontent.com/103648207/174852206-2bf930da-ae4c-4293-bb1a-7818eaa1ab00.png">
<img width="615" alt="Screenshot 2022-06-21 at 17 35 26" src="https://user-images.githubusercontent.com/103648207/174852319-342c0405-ee32-453c-bb2d-09981d645493.png">

# Datasets used

<img width="177" alt="Screenshot 2022-06-22 at 16 44 04" src="https://user-images.githubusercontent.com/103648207/175074550-c72df250-b26a-4974-81af-759467e95958.png">

ImageNet - image database designed for use in computer vision research

# Output predictions

<img width="743" alt="Screenshot 2022-06-22 at 16 50 16" src="https://user-images.githubusercontent.com/103648207/175076032-d6483be1-cd86-45a7-9b9c-51cb41dd5ab7.png">


# Performance metrics

- Doc2vec for transfering sentences to vectors

- Cosine similarities as scores
 
<img width="376" alt="Screenshot 2022-06-22 at 16 13 51" src="https://user-images.githubusercontent.com/103648207/175067109-e4a1c8e4-5a75-4bc5-835b-785c377e1e57.png">

# Final product

- Link to Streamlit : [https://awesome-github-readme-profile.netlify.app](https://share.streamlit.io/cmaxk/robo_romeo_streamlit/app.py)
- Link to Streamlit GitHub : https://github.com/CMaxK/robo_romeo_streamlit
- Link to demo presentation slides : https://docs.google.com/presentation/d/19MzJlfLe1qM_8c3-CEjDwYxT5BYAXQVz09pqFLd45gA/edit#slide=id.g134fb78e839_0_201

# Our Robo-Romeo's Output

<img width="1715" alt="Screenshot 2022-06-21 at 17 24 45" src="https://user-images.githubusercontent.com/103648207/174849984-cfd70617-4a2f-498d-b9e8-ec978ce8d439.png">

<img width="1717" alt="Screenshot 2022-06-21 at 17 44 47" src="https://user-images.githubusercontent.com/103648207/174853791-b04d34f6-3e49-41ca-ace8-51138948287b.png">


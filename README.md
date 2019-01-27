# Deep Audio


## Setup Instructions
1. Follow the installation instructions for a tensorflow docker container: https://www.tensorflow.org/install/docker#tensorflow_docker_requirements
2. When creating the docker container run the following to create a persistant container: 
    ```bash
    sudo docker container create \
        --name deep-audio \
        --restart always \
        -p 8888:<exposed_port> \
        -p 6006:<exposed_port> \
        tensorflow/tensorflow:latest-gpu-py3
    ```
3. Once created, be sure that the port you use for jupyter notebooks is open on the server (by default 8888)
4. Open a bash shell into your container using
    ```bash
    sudo docker exec -it deep-audio /bin/bash
    ```

5. Install dependencies:

    ```bash
    apt update
    apt install git openssh-server openssh-client nano libsndfile-dev
    ```
6. Follow add ssh key to Github:
   
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa
   cat ~/.ssh/id_rsa.pub
   ```
   >From:   https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/#platform-linux

7. Clone the repository into the docker container
    ```bash
    git clone --recursive git@github.com:georgia-tech-db/deep-audio.git
    ```

8. Enter the directory and install pip dependencies
    ```
    pip install -r requirements.txt
    ```

9. (Optional) Exit the container and if remote reconnect to the server with ssh tunneling:
    ```
    ssh -L 8888:localhost:8888 usr@server
    ```
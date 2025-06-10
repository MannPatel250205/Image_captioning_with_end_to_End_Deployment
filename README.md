# AZURE-CICD-Deployment-with-Github-Actions

## Save pass:




## Run from terminal:

docker build -t imagecaptionendtoend.azurecr.io/chicken:latest .

docker login imagecaptionendtoend.azurecr.io

docker push imagecaptionendtoend.azurecr.io/chicken:latest


## Deployment Steps:

1. Build the Docker image of the Source Code
2. Push the Docker image to Container Registry
3. Launch the Web App Server in Azure 
4. Pull the Docker image from the container registry to Web App server and run 
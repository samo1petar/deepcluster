docker run -it --runtime=nvidia --name deepcluster_cont \
    --mount type=bind,source='/path/to/data/like/Flickr25K',target='/home/data' \
    deepcluster /bin/bash
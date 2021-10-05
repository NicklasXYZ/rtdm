
# Pull the official python 3.6 base image.
FROM python:3.6

# Create a new directory for user uploaded and generated data.
# (https://www.pathname.com/fhs/pub/fhs-2.3.html)
RUN mkdir -p /srv/backend/filestorage

# Set the current working directory.
WORKDIR /usr/src/backend

# Set container environment variables (also available after the build process).
ENV PYTHONUNBUFFERED = 1
ENV PYTHONDONTWRITEBYTECODE = 1

# Update pip
RUN pip install --upgrade pip

# Copy the python dependency file into the current working directory.
COPY ./req.txt /usr/src/backend/req.txt

# Install all required python libraries.
RUN pip install -r req.txt

# Copy the project directory, that contains all the source code, into the docker container.
COPY ./backend /usr/src/backend
COPY ./tsanomdet /usr/src/backend

# Make sure the entrypoint.sh script is executable.
RUN chmod +x /usr/src/SOD/entrypoint.sh

# Run the entrypoint.sh script. Always run this whenever a new docker container is set up.
ENTRYPOINT ["/usr/src/SOD/entrypoint.sh"]

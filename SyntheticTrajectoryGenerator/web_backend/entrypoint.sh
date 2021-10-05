#!/bin/sh
#python manage.py makemigrations backend
#python manage.py migrate
#echo "Migrations were run successfully!"
# Execute whatever command was passed to the container
# via the docker-compose config. file
exec "$@"

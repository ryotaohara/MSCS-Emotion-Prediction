FROM mysql:latest

# Set root credentials (can be overridden at runtime too)
ENV MYSQL_ROOT_PASSWORD rootpass
ENV MYSQL_DATABASE emotion_db
#ENV MYSQL_USER user
#ENV MYSQL_PASSWORD pass

# Copy init script into the initialization directory
COPY init-db.sql /docker-entrypoint-initdb.d/

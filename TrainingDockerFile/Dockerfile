# Use an official Java runtime as a parent image
FROM openjdk:11-jre-slim

# Set the working directory in the container
WORKDIR /app

# Copy the JAR file into the container at /app
COPY target/TrainingClass-1.0-SNAPSHOT.jar /app/target/TrainingClass-1.0-SNAPSHOT.jar

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV JAVA_OPTS=""

# Run the application
ENTRYPOINT ["java", "-jar", "TrainingClass-1.0-SNAPSHOT.jar"]

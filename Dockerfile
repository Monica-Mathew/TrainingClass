#Defines a base for your image.
FROM eclipse-temurin:11-jdk

# Sets the working directory
WORKDIR /app

# Copy the JAR file into the container at /app
COPY target/TrainingClass-1.0-SNAPSHOT.jar /app/TrainingClass-1.0-SNAPSHOT.jar
COPY TrainingDataset.csv /app/TrainingDataset.csv
COPY ValidationDataset.csv /app/ValidationDataset.csv

# The EXPOSE instruction marks that our final image has a service listening on port 8000.
EXPOSE 8080

# Define environment variable
ENV JAVA_OPTS=""

# Run the application
ENTRYPOINT ["java", "-jar", "TrainingClass-1.0-SNAPSHOT.jar"]
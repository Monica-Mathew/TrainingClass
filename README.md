Cloud Computing
Programming assignment

Wine Quality Training Module over AWS EMR

#### Building the project
```
mvn clean package
```


#### Running the project
```
spark-submit --class org.example.WineQualityTraining --master 'local[*]' target/original-TrainingClass-1.0-SNAPSHOT.jar
```

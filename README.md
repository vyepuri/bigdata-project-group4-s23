
## Big Data Project - Group 5 - Spring 2023



<br/><br/>
### Step 1 : *Load data into hive*

#### After logging in, clone the code and requred files
```sh
>> cd ~
>> git clone https://github.com/vyepuri/bigdata-project-group5-s23.git
```

#### Connect to hive

```sh
>> hive
```

#### Create table in hive with name sales 
```sh
hive>> CREATE TABLE sales (
  invoice_id STRING,
  branch STRING,
  city STRING,
  customer_type STRING,
  gender STRING,
  product_line STRING,
  unit_price FLOAT,
  quantity INT,
  tax FLOAT,
  total FLOAT,
  purchase_date STRING,
  purchase_time STRING,
  payment STRING,
  cogs FLOAT,
  gross_margin_percentage FLOAT,
  gross_income FLOAT,
  rating FLOAT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;
```
#### View tables in the default database
```sh
hive>> show tables;
```
#### Load csv data into hive table
```sh
hive>> LOAD DATA LOCAL INPATH '/root/bigdata-project-group5-s23/dataset/sales.csv' OVERWRITE INTO TABLE sales;
```

#### Query the data and verify
```sh
hive>> select * from sales;
```

#### Come out of hive shell
```sh
hive>> exit;
```

<br/><br/>




### Step 2 : *Run the interactive Visualizer application*

#### Setup libraries
```sh
>> pip3 install flask, plotly
>> sudo apt install unixodbc-dev
>> sudo apt-get install libsasl2-dev
>> pip3 install sasl
```

#### Run the application
```sh
>> cd /root/bigdata-project-group5-s23/code
>> python app.py
```


#### Access the application on browser with ip:5000

  
  <br/><br/>

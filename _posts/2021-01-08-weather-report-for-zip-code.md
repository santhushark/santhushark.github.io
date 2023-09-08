---
layout: post
title: Weather Report for a given Zip Code
description: "A Backend Springboot Application to get the weather report for a given Zip Code using Google Geo Codes API"
author: santhushark
category: Web Application
tags: springboot backend-application java google-geo-codes-api 
finished: true
---
## Introduction


The aim of the project is to build a Springboot backend Java Application to get weather report for a given ZIP code using Google Geocoding API's and Open Weather Map API's.

What is Springboot?

Java Spring Framework (Spring Framework) is a popular, open source, enterprise-level framework for creating standalone, production-grade applications that run on the Java Virtual Machine (JVM).
Java Spring Boot (Spring Boot) is a tool that makes developing web application and microservices with Spring Framework faster and easier through three core capabilities:
- Autoconfiguration
- An opinionated approach to configuration
- The ability to create standalone applications

These features work together to provide you with a tool that allows you to set up a Spring-based application with minimal configuration and setup.

What is Google Geocoding API?

The Geocoding API is a service that accepts a place as an address, latitude and longitude coordinates, or Place ID. It converts the address into latitude and longitude coordinates and a Place ID, or converts latitude and longitude coordinates or a Place ID into an address.

Use the Geocoding API for website or mobile application when you want to use geocoding data within maps provided by one of the Google Maps Platform APIs. With the Geocoding API, you use addresses to place markers on a map, or convert a marker on a map to an address. This service is designed for geocoding predefined, static addresses for placement of application content on a map.

What is Open Weather Map?

OpenWeatherMap is an online service, owned by OpenWeather Ltd, that provides global weather data via API, including current weather data, forecasts, nowcasts and historical weather data for any geographical location. The company provides a minute-by-minute hyperlocal precipitation forecast for any location.

## Database


Mysql database is used in the project. Run the [DBscript](https://github.com/santhushark/Weather-Report/blob/main/weather/dbscripts/script_14_nov_2020.sql) to create database and the necessary tables.

## Source code

MVC architecture is followed in the project i.e. The controller layer which is the entry point for endpoints. Service Layer which is where the business logic is written and the Model layer where the different tables are defined.

#### Controller

[WeatherController](https://github.com/santhushark/Weather-Report/blob/main/weather/webservices/src/main/java/com/assignment/weather/controller/WeatherController.java) has the REST endpoint which takes in ZIPCODE, CITY, COUNTRY CODE and returns Weather Details of that particular region. The weather details include **temperature, min temperature, max temperature, humudity and windspeed** 

#### Services

Here we have three main services i.e. [HttpRestTemplateService](https://github.com/santhushark/Weather-Report/blob/main/weather/webservices/src/main/java/com/assignment/weather/service/HttpRestTemplateService.java), [WeatherApiService](https://github.com/santhushark/Weather-Report/blob/main/weather/webservices/src/main/java/com/assignment/weather/service/WeatherApiService.java) and [WeatherDetailsService](https://github.com/santhushark/Weather-Report/blob/main/weather/webservices/src/main/java/com/assignment/weather/service/WeatherDetailsService.java).

HttpRestTemplateService basically handles the http request to third party API's, in this case its Google Geocoding API's and Openweathermap API's. Initially with ZIPCODE we get the Latitude and Longitude details using Geocoding API's and then the Lat Long values are sent with Openweathermap API's to get the weather details. The weather report response is then processed in WeatherApiService and WeatherDetailsService and stored in the Database and the response is constructed and sent back to frontend.

#### Models

There are two tables, The **geo_code** stores the Latitude and Longitude values for a given Zip Code so that when there are requests to get the weather report for the same ZIPCODE the lat long values are fetched from database and not from Geocoding API's. The **wthr_rprt** table stores the weather report of a particular ZIPCODE.

## ENDPOINT


API to test from Postman or any online Rest API testing tool:

***localhost:8080/weather/todayWeather/zipCode=560061/city=Bengaluru/countryCode=IN***

It is localhost, since the application is run on a local machine. Aslo replace YOUR_GOOGLE_API_KEY and YOUR_OPEN_WEATHER_MAP_API_KEY in [application.properties](https://github.com/santhushark/Weather-Report/blob/main/weather/webservices/src/main/resources/application.properties) in webservices with your API keys before running the application.



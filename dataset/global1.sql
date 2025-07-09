use global;

select * from global1;

select count(*) as total_records,
sum( case when `customer id` is null or `customer id`='' then 1 else 0 end) AS missing_1,
sum( case when `customer name` is null or `customer name`='' then 1 else 0 end) AS missing_2,
sum( case when `ship mode` is null or `ship mode`='' then 1 else 0 end) AS missing_3,
sum( case when `product id` is null or `product id`='' then 1 else 0 end) AS missing_4,
sum( case when `category` is null or `category`='' then 1 else 0 end) AS missing_5,
sum( case when `sub-category` is null or `sub-category`='' then 1 else 0 end) AS missing_6,
sum( case when `product name` is null or `product name`='' then 1 else 0 end) AS missing_7,
sum( case when `sales` is null or `sales`='' then 1 else 0 end) AS missing_8
from global1;

SELECT `order date`, COUNT(*) AS Count
FROM global1
GROUP BY `order date`
ORDER BY Count DESC;


SELECT YEAR(STR_TO_DATE(`Order Date`, '%d/%m/%y')) AS Year, Region, SUM(Sales) AS Total_Sales
FROM global1
GROUP BY Year, Region
ORDER BY Year,region,total_sales desc;

SELECT 
    YEAR(STR_TO_DATE(`Order Date`, '%d/%m/%y')) AS Year,
    Region,
    Sales,
    SUM(Sales) OVER (
        PARTITION BY Region, YEAR(STR_TO_DATE(`Order Date`, '%d/%m/%y')) 
        ORDER BY STR_TO_DATE(`Order Date`, '%d/%m/%y')
    ) AS Cumulative_Sales
FROM global1
ORDER BY Region, Year, STR_TO_DATE(`Order Date`, '%d/%m/%y');


SELECT YEAR(STR_TO_DATE(`Order Date`, '%d/%m/%y')) AS Year, Region, SUM(Sales) AS Total_Sales,
       SUM(SUM(Sales)) OVER (PARTITION BY Region ORDER BY YEAR(STR_TO_DATE(`Order Date`, '%d/%m/%y'))) AS Cumulative_Sales
FROM global1
GROUP BY Year, Region;

SELECT State, City, SUM(Sales) AS Total_Sales,
       RANK() OVER (PARTITION BY State ORDER BY SUM(Sales) DESC) AS Sales_Rank
FROM global1
GROUP BY State, City;


WITH SalesData AS (
    SELECT YEAR(STR_TO_DATE(`Order Date`, '%d/%m/%y')) AS Year, Region, SUM(Sales) AS Total_Sales
    FROM global1
    GROUP BY Year, Region
)
SELECT *,
       LAG(Total_Sales) OVER (PARTITION BY Region ORDER BY Year) AS Prev_Year_Sales,
       ROUND(((Total_Sales - LAG(Total_Sales) OVER (PARTITION BY Region ORDER BY Year)) / LAG(Total_Sales) OVER (PARTITION BY Region ORDER BY Year)) * 100, 2) AS YoY_Growth
FROM SalesData;


WITH ProductSales AS (
    SELECT Category, `Product Name`, SUM(Sales) AS Total_Sales,
           ROW_NUMBER() OVER (PARTITION BY Category ORDER BY SUM(Sales) DESC) AS Product_Rank
    FROM global1
    GROUP BY Category, `Product Name`
)
SELECT * FROM ProductSales WHERE Product_Rank <= 3;


WITH StateAvg AS (
    SELECT State, AVG(Sales) AS Avg_Sales
    FROM global1
    GROUP BY State
),
NationalAvg AS (
    SELECT AVG(Sales) AS National_Avg FROM global1
)
SELECT s.State, s.Avg_Sales
FROM StateAvg s, NationalAvg n
WHERE s.Avg_Sales > n.National_Avg;


WITH CustomerSales AS (
    SELECT `Customer ID`, SUM(Sales) AS Total_Sales
    FROM global1
    GROUP BY `Customer ID`
)
SELECT `Customer ID`, Total_Sales,
       CASE
           WHEN Total_Sales >= 10000 THEN 'High Spender'
           WHEN Total_Sales >= 5000 THEN 'Medium Spender'
           ELSE 'Low Spender'
       END AS Customer_Category
FROM CustomerSales
ORDER BY Total_Sales DESC;

SELECT YEAR(STR_TO_DATE(`Order Date`, '%d/%m/%y')) AS Year, Segment, SUM(Sales) AS Total_Sales
FROM global1
GROUP BY Year, Segment
ORDER BY Year, Segment;

WITH RegionalSales AS (
    SELECT Region, SUM(Sales) AS Regional_Total
    FROM global1
    GROUP BY Region
)
SELECT g.State, g.Region, SUM(g.Sales) AS State_Total,
       r.Regional_Total,
       ROUND((SUM(g.Sales) / r.Regional_Total) * 100, 2) AS State_Percentage_Contribution
FROM global1 g
JOIN RegionalSales r ON g.Region = r.Region
GROUP BY g.State, g.Region, r.Regional_Total
ORDER BY g.Region, State_Percentage_Contribution DESC;

WITH MonthlySales AS (
    SELECT Region, DATE_FORMAT(STR_TO_DATE(`Order Date`, '%d/%m/%y'), '%Y-%m') AS Order_Month, SUM(Sales) AS Total_Sales
    FROM global1
    GROUP BY Region, Order_Month
)
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY Region ORDER BY Total_Sales DESC) AS Rank_High,
           ROW_NUMBER() OVER (PARTITION BY Region ORDER BY Total_Sales ASC) AS Rank_Low
    FROM MonthlySales
) ranked
WHERE Rank_High = 1 OR Rank_Low = 1
ORDER BY Region, Rank_High, Rank_Low;

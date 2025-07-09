SELECT * FROM churn.`wa_fn-usec_-telco-customer-churn`;

rename table  `wa_fn-usec_-telco-customer-churn` to churns;

select * from churns;

select count(*) as total_no from churns;

select count(*) as churned_customers from churns where churn='yes';


select churn,count(*) as total,round(count(*) * 100 /(select count(*) from churns),2) as rate
from churns 
where churn='yes';


select churn,count(*) as total,round(count(*) * 100 /(select count(*) from churns),2) as rate
from churns 
group by churn;

select customerid,contract from churns
group by customerid,contract
order by contract;


WITH ranked_customers AS (
    SELECT 
        customerid,
        contract,
        ROW_NUMBER() OVER (PARTITION BY contract ORDER BY RAND()) AS rn
    FROM churns
)
SELECT customerid, contract
FROM ranked_customers
WHERE rn <= 10
ORDER BY contract, rn;


select total_month_month,rate
  from churns where contract='month-to-month';
  
  with rate as (
select 
  contract,
  count(*) as total,
  round(count(*) * 100 /(select count(*) from churns),2) as rate
  from churns
  group by contract
  )
  
  select contract,rate,total
  from rate
  order by rate desc;
  
  
  SELECT 
    COUNT(*) AS Total_Customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS Churn_Rate_Percentage
FROM Churns;


select contract, sum( case when churn='yes' then 1 else 0 end) as no_of_churns,
		round((sum(case when churn='yes' then 1 else 0 end) * 100)/count(*),2) as rate
        from churns
        group by contract
        order by rate desc;
        
select 
		case when tenure between 0 and 12 then '0 - 12 months'
			 when tenure between 12 and 24 then '1 - 2 years'
             when tenure between 24 and 36 then '2 - 3 years'
             else '37+ months'
             end as tenure_segments,
             count(*) as total_customers,
             sum(case when churn='yes' then 1 else 0 end) as churned_customers,
             round((sum(case when churn='yes' then 1 else 0 end)*100)/count(*),2) as churn_rate
             from churns
             group by tenure_segments
             order by churn_rate desc;
             
select internetservice,
			   round((sum(case when churn='yes' then 1 else 0 end)* 100)/count(*),2)as rate
               from churns
               group by internetservice
			   order by rate desc;
               

SELECT 
    churn,
    ROUND(count(*) *100/(select count(*) from churns),2) as rate,
    ROUND(SUM(totalcharges), 0) AS total_charge
FROM churns
GROUP BY churn
ORDER BY total_charge desc;


SELECT 
    CASE 
        WHEN MonthlyCharges <= 50 THEN 'Low (<=50)'
        WHEN MonthlyCharges BETWEEN 51 AND 100 THEN 'Medium (51-100)'
        ELSE 'High (>100)'
    END AS Charge_Bucket,
    COUNT(*) AS Total_Customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS Churn_Rate_Percentage
FROM churns
GROUP BY Charge_Bucket
ORDER BY Churn_Rate_Percentage DESC;


SELECT 
    SeniorCitizen,
    Dependents,
    COUNT(*) AS Total_Customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS Churn_Rate_Percentage
FROM Churns
GROUP BY SeniorCitizen, Dependents
ORDER BY Churn_Rate_Percentage DESC;



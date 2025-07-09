use data;

select * from delhivery;

select status,count(*) as delivery_count,
round(count(*) * 100.0 / (select count(*) from delhivery),2) as percentage
from delhivery
group by status;

with route_stats as (
select route_type,
		count(*) as total_deliveries,
        sum(case when status ='delayed' then 1 else 0 end) as delayed_deliveries
        from delhivery
        group by route_type
        )
        SELECT 
    route_type,
    total_deliveries,
    delayed_deliveries,
    ROUND(delayed_deliveries * 100.0 / total_deliveries, 2) AS delay_rate
FROM route_stats
ORDER BY delay_rate DESC
LIMIT 5;



select route_type,
		total_deliveries,
         percentage,
        round(percentage * 100/total_deliveries,2) as delivery_rate
        from (
        select route_type,
				count(*) as total_deliveries,
                sum(case when status='delayed' then 1 else 0 end) as percentage 
                from delhivery
                group by route_type
                order by percentage desc)as nall
                ;
                
                
                
WITH delays AS (
    SELECT 
        time_of_day,
        COUNT(*) AS counts,
        SUM(CASE WHEN status = 'delayed' THEN 1 ELSE 0 END) AS delivery_status_of_each
    FROM DELHIVERY
    GROUP BY time_of_day
)
SELECT 
    time_of_day,
    counts,
    delivery_status_of_each
FROM delays; 

SELECT 
    TRIM(SUBSTRING_INDEX(`day&time`, ',', 1)) AS day_of_week,
    COUNT(*) AS total_deliveries,
    SUM(CASE WHEN status = 'delayed' THEN 1 ELSE 0 END) AS total_delays
FROM delhivery
GROUP BY day_of_week
ORDER BY total_delays DESC;

SELECT 
    day_of_week,
    total_deliveries,
    SUM(total_deliveries) OVER (ORDER BY FIELD(day_of_week, 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')) AS cumulative_deliveries
FROM (
    SELECT 
        TRIM(SUBSTRING_INDEX(`day&time`, ',', 1)) AS day_of_week,
        COUNT(*) AS total_deliveries
    FROM delhivery
    GROUP BY day_of_week
) AS daily_deliveries
ORDER BY FIELD(day_of_week, 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun') desc;


SELECT 
    source_name,
    ROUND(AVG(delivery_delay_time), 2) AS avg_delay
FROM delhivery
GROUP BY source_name
HAVING avg_delay > (SELECT AVG(delivery_delay_time) FROM delhivery)
ORDER BY avg_delay DESC;


SELECT 
    ROUND(actual_distance_to_destination, 4) AS rounded_distance,
    delivery_delay_time
FROM delhivery
ORDER BY actual_distance_to_destination;
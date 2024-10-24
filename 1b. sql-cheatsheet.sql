
---------------------------------------------------------------------
-- DATABASE INFO
---------------------------------------------------------------------

-- ENTITY DEFINITIONS
-- Entity: Represents a real-world object or concept in a database.
-- Example Entities: Customers, Orders, Products, Students, Courses, Employees, etc.
        CREATE TABLE Users (
            UserID INT PRIMARY KEY,
            UserName VARCHAR(100)
        );

        CREATE TABLE UserDetails (
            UserID INT PRIMARY KEY,
            Address VARCHAR(255),
            FOREIGN KEY (UserID) REFERENCES Users(UserID) ON DELETE SET NULL -- One-to-One relationship
        );

-- ATTRIBUTE DEFINITIONS
-- Attributes: Descriptive properties of an entity.
-- Example Attributes for an "Employee" Entity: EmployeeID, Name, DepartmentID, HireDate, Salary.

-- RELATIONSHIPS
-- Relationships define how entities are connected to each other:
-- One-to-One (1:1): A single record in Table A corresponds to a single record in Table B.
-- One-to-Many (1:M): A single record in Table A corresponds to multiple records in Table B.
-- Many-to-Many (M:N): Multiple records in Table A correspond to multiple records in Table B.

-- NOTES
-- - Primary Keys ensure data uniqueness and integrity. (types include simple primary key, composite, natural, and surrogate primary keys)
        ProductID INT PRIMARY KEY,
        PRIMARY KEY (ProductID) 
-- Composite Key: Combination of two or more columns to uniquely identify records.
        PRIMARY KEY (OrderID, ProductID) 
-- - Foreign Keys enforce relationships between tables.
        FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
-- UNIQUE Key: Ensures all values in a column are unique.
        SKU VARCHAR(50) UNIQUE  -- Each SKU must be unique
-- - Relationships can be established using primary keys and foreign keys.
-- - Consider adding UNIQUE constraints for columns that must be unique.



---------------------------------------------------------------------
-- Constraints in SQL
---------------------------------------------------------------------

PRIMARY KEY     -- Uniquely identifies each record in a table
FOREIGN KEY     -- Establishes a relationship between tables
UNIQUE          -- Ensures all values in a column are unique
NOT NULL        -- Ensures that a column cannot have NULL values
CHECK           -- Ensures that all values in a column satisfy a specific condition
DEFAULT         -- Sets a default value for a column if no value is specified
AUTO_INCREMENT      -- Automatically generates unique values (commonly used in MySQL)
IDENTITY            -- Automatically generates unique values (commonly used in SQL Server)
INDEX               -- Creates an index to allow faster retrieval of records
CASCADE             -- Automatically propagates changes in a foreign key to the related records
ON DELETE CASCADE   -- Deletes related records when a parent record is deleted
ON UPDATE CASCADE   -- Updates related records when a parent record is updated
UNIQUEIDENTIFIER    -- Globally unique identifier constraint (using `NEWID()` in SQL Server)
PRIMARY KEY with CLUSTERED INDEX -- Organizes table data based on the primary key (default in SQL Server)
DEFAULT with NEWID()-- Generates a globally unique value for a column by default (when used with UNIQUEIDENTIFIER in SQL Server)
;


---------------------------------------------------------------------
-- SQL DATA TYPES
---------------------------------------------------------------------

-- Numeric Data Types
INT, BIGINT, SMALLINT, TINYINT  -- Whole numbers of various sizes
DECIMAL(p,s), NUMERIC(p,s)      -- Fixed-point numbers (p: precision, s: scale)
FLOAT, REAL                     -- Floating-point numbers
BIT                             -- Boolean values (0 or 1)

-- Date and Time Data Types
DATE                            -- Only date (YYYY-MM-DD)
DATETIME                        -- Date and time (YYYY-MM-DD HH:MM:SS)
DATETIME2                       -- More precise datetime
TIME                            -- Only time (HH:MM:SS)
DATETIMEOFFSET                  -- Date, time, and timezone

-- String Data Types
CHAR(n), VARCHAR(n)             -- Fixed and variable-length strings
VARCHAR(MAX)                    -- Variable-length large text
NCHAR(n), NVARCHAR(n)           -- Fixed and variable-length Unicode strings
NVARCHAR(MAX)                   -- Variable-length large Unicode text

-- Binary Data Types
BINARY(n), VARBINARY(n)         -- Fixed and variable-length binary data
VARBINARY(MAX)                  -- Variable-length large binary data

-- Special Data Types
XML                             -- Stores XML formatted data
UNIQUEIDENTIFIER                -- Globally unique identifier (GUID)
SQL_VARIANT                     -- Stores values of different data types
JSON (stored as NVARCHAR)       -- JSON data stored in NVARCHAR columns


;---------------------------------------------------------------------
-- Get all Database Tables
---------------------------------------------------------------------

SELECT * FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_TYPE = 'BASE TABLE';
;---------------------------------------------------------------------
-- List of Most Used SQL Commands
---------------------------------------------------------------------

-- Data Retrieval Commands:
    SELECT -- Extracts data from a database.
        SELECT column1, column2 FROM table_name WHERE condition; 
    SELECT DISTINCT -- Returns unique records
        SELECT DISTINCT column_name FROM table_name;
    SELECT COUNT (column1) FROM table_name WHERE sex = 'F' AND birth_date > '1971-01-01'; -- counts all the items in column1
        -- COUNT: counts all the items in column1
        -- SUM: calculates the total sum of column1
        -- AVG: calculates the average value of column1
        -- MIN: finds the smallest value in column1
        -- MAX: finds the largest value in column1
        -- VARIANCE: calculates the variance of column1 (use VAR in SQL Server)
        -- STDDEV: calculates the standard deviation of column1 (use STDEV in SQL Server)
        -- GROUP_CONCAT: concatenates values from multiple rows into a single string (MySQL), or use STRING_AGG in SQL Server/PostgreSQL
        -- COUNT(DISTINCT): counts the number of unique values in column1
        -- MEDIAN: calculates the median value of column1 (available in PostgreSQL using percentile functions)
        -- FIRST: returns the first value in a set of rows (available in PostgreSQL and some SQL dialects)
        -- LAST: returns the last value in a set of rows (available in PostgreSQL and some SQL dialects)
        -- MODE: returns the most frequently occurring value in column1 (available in PostgreSQL)

    WHERE -- Filters records based on a condition.
        SELECT column1, column2 FROM table_name WHERE condition;
        -- Example conditions: column1 = value, column2 > value, column3 LIKE '%text%'
        -- <, >, <=, >=, <> (means not equal to), AND, OR, LIKE '%text%', etc.  
    ORDER BY -- Sorts the result set.
        SELECT column1 FROM table_name ORDER BY column1 ASC; --ASC or DSC is ascending and descending
    GROUP BY -- Groups rows that have the same values in specified columns.
        SELECT column1, COUNT(*) FROM table_name GROUP BY column1;
    HAVING -- Filters groups based on a condition.
        SELECT column1, COUNT(*) FROM table_name GROUP BY column1 HAVING COUNT(*) > 1;
    LIMIT -- Limits the number of returned rows (MySQL/PostgreSQL).
        SELECT * FROM table_name LIMIT 10;
    TOP -- Limits the number of returned rows (SQL Server).
        SELECT TOP 5 * FROM table_name;
    OFFSET -- Skips a specific number of rows before returning the result (commonly used with LIMIT).
        SELECT column1 FROM table_name ORDER BY column1 OFFSET 10 ROWS;

    BETWEEN -- Filters data within a range of values.
        SELECT * FROM table_name WHERE column1 BETWEEN value1 AND value2;

    LIKE -- Searches for a specified pattern in a column.
        SELECT * FROM table_name WHERE column_name LIKE 'pattern';  -- Example patterns: 'A%', '%B', '_C%'

    IN -- Filters rows based on a list of values.
        SELECT * FROM table_name WHERE column_name IN (value1, value2, ...);

    EXISTS -- Checks if a subquery returns any records.
        SELECT * FROM table_name WHERE EXISTS (SELECT column_name FROM another_table WHERE condition);

    ALL -- Compares a value to all values in a subquery.
        SELECT column_name FROM table_name WHERE column_name > ALL (SELECT column_name FROM another_table);

    ANY -- Compares a value to any value in a subquery.
        SELECT column_name FROM table_name WHERE column_name = ANY (SELECT column_name FROM another_table);

    UNION -- Combines the result sets of two or more SELECT statements (removes duplicates).
        SELECT column1 FROM table_name1 UNION SELECT column1 FROM table_name2;

    UNION ALL -- Combines the result sets of two or more SELECT statements (includes duplicates).
        SELECT column1 FROM table_name1 UNION ALL SELECT column1 FROM table_name2;

    INTERSECT -- Returns only the records that are present in both SELECT statements.
        SELECT column1 FROM table_name1 INTERSECT SELECT column1 FROM table_name2;

    EXCEPT (or MINUS) -- Returns records from the first SELECT statement that are not present in the second.
        SELECT column1 FROM table_name1 EXCEPT SELECT column1 FROM table_name2;  -- In MySQL, use MINUS

    JOIN -- Combines rows from two or more tables based on a related column.
        SELECT A.column1, B.column2 FROM table1 A JOIN table2 B ON A.id = B.id;

    INNER JOIN -- Returns only the rows with matching values in both tables.
        SELECT A.column1, B.column2 FROM table1 A INNER JOIN table2 B ON A.id = B.id;

    LEFT JOIN (or LEFT OUTER JOIN) -- Returns all rows from the left table, and matched rows from the right table.
        SELECT A.column1, B.column2 FROM table1 A LEFT JOIN table2 B ON A.id = B.id;

    RIGHT JOIN (or RIGHT OUTER JOIN) -- Returns all rows from the right table, and matched rows from the left table.
        SELECT A.column1, B.column2 FROM table1 A RIGHT JOIN table2 B ON A.id = B.id;

    FULL JOIN (or FULL OUTER JOIN) -- Returns all rows when there is a match in either left or right table.
        SELECT A.column1, B.column2 FROM table1 A FULL JOIN table2 B ON A.id = B.id;

    CROSS JOIN -- Returns all possible combinations of rows from both tables.
        SELECT A.column1, B.column2 FROM table1 A CROSS JOIN table2 B;

    SELF JOIN -- Joins a table to itself to find related rows within the same table.
        SELECT A.column1, B.column2 FROM table_name A, table_name B WHERE A.id = B.parent_id;

    CASE -- Provides conditional logic to return specific values based on conditions.
        SELECT column1,
            CASE 
                WHEN condition1 THEN 'Result1'
                WHEN condition2 THEN 'Result2'
                ELSE 'DefaultResult'
            END AS AliasName
        FROM table_name;



-- Data Modification Commands:
    INSERT INTO -- Inserts new data into a database.
        INSERT INTO table_name (column1, column2) VALUES (value1, value2);
    UPDATE -- Updates data in a database.
        UPDATE table_name SET column1 = value1 WHERE condition;
    DELETE -- Deletes data from a database.
        DELETE FROM table_name WHERE condition;



-- Data Definition Commands:
    CREATE TABLE -- Creates a new table.
        CREATE TABLE table_name (
            column1 datatype PRIMARY KEY,
            column2 datatype,
            ...
        );
    AUTO_INCREMENT / IDENTITY --- Automatically generates unique values for a column.
        -- MySQL
        CREATE TABLE Customers (
            CustomerID INT AUTO_INCREMENT PRIMARY KEY,
            CustomerName VARCHAR(100)
        );

        -- SQL Server
        CREATE TABLE Orders (
            OrderID INT IDENTITY(1,1) PRIMARY KEY,
            OrderDate DATE
        );
    ALTER TABLE -- Modifies an existing table.
        ALTER TABLE table_name ADD column_name datatype;
    DROP TABLE -- Deletes a table.
        DROP TABLE table_name;




-- Indexing and Constraints:
    CREATE INDEX -- Creates an index for faster retrieval.
        CREATE INDEX idx_column ON table_name (column);
    UNIQUE -- Ensures all values in a column are unique.
        CREATE TABLE Products (
            ProductID INT PRIMARY KEY,
            SKU VARCHAR(50) UNIQUE
        );
    PRIMARY KEY -- Uniquely identifies each record in a table.
        CREATE TABLE Employees (
            EmployeeID INT PRIMARY KEY,
            Name VARCHAR(100)
        );
    FOREIGN KEY -- Establishes a relationship between two tables.
        CREATE TABLE Orders (
            OrderID INT PRIMARY KEY,
            CustomerID INT,
            FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
        );




-- Data Control and Transaction Commands:
    GRANT -- Grants permissions to a user.
        GRANT SELECT, INSERT ON table_name TO user_name;
    REVOKE -- Revokes permissions from a user.
        REVOKE INSERT ON table_name FROM user_name;
    COMMIT -- Saves the current transaction.
        COMMIT;
    ROLLBACK -- Reverts the current transaction.
        ROLLBACK;
    SAVEPOINT -- Creates a savepoint within a transaction.
        SAVEPOINT SavepointName;



-- Views and Other Constructs:
    -- CREATE VIEW -- Creates a virtual table based on the result of a SELECT query.
    --     CREATE VIEW ActiveUsers AS
    --     SELECT UserID, UserName FROM Users WHERE IsActive = 1;
    ;
    -- CREATE PROCEDURE -- Creates a stored procedure.
    --     CREATE PROCEDURE GetOrders
    --     AS
    --     BEGIN
    --         SELECT * FROM Orders;
    --     END;
    ;
    -- CREATE FUNCTION -- Creates a user-defined function.
    --     CREATE FUNCTION GetUserName (@UserID INT) RETURNS VARCHAR(100)
    --     AS
    --     BEGIN
    --         DECLARE @UserName VARCHAR(100);
    --         SELECT @UserName = UserName FROM Users WHERE UserID = @UserID;
    --         RETURN @UserName;
    --     END;
    ;
    -- CREATE TRIGGER -- Creates a trigger that automatically executes code based on specific events.
    --     CREATE TRIGGER trg_AfterInsert
    --     ON Orders
    --     AFTER INSERT
    --     AS
    --     BEGIN
    --         INSERT INTO AuditLog (OrderID, LogDate) 
    --         SELECT OrderID, GETDATE() FROM inserted;
    --     END;
;



-- Utilities and Information:
    SHOW TABLES -- Lists all tables in a database (MySQL).
    SHOW DATABASES -- Lists all databases (MySQL).
    DESCRIBE -- Provides details about a table structure (MySQL/PostgreSQL) or sp_help in SQL Server.


-- Generating Unique Identifiers:
    NEWID() -- Generates a globally unique identifier (GUID).
        CREATE TABLE Products (
            ProductID UNIQUEIDENTIFIER DEFAULT NEWID() PRIMARY KEY,
            ProductName VARCHAR(100)
        );







---------------------------------------------------------------------
-- Additional Important SQL Commands
---------------------------------------------------------------------
;
-- BASIC SELECT Statement
SELECT select_list
[ FROM table_source ]
[ WHERE search_condition ]
[ GROUP BY group_by_expression ]
[ HAVING search_condition ]
[ ORDER BY order_expression [ ASC | DESC ] ];



-- WHERE Conditions
-- AND / OR
SELECT column_name 
FROM table_name
WHERE condition1 AND condition2;

SELECT column_name 
FROM table_name
WHERE condition1 OR condition2;


-- EXISTS
SELECT column_name 
FROM table_name
WHERE EXISTS (SELECT column_name FROM table_name WHERE condition);

-- ANY / ALL
SELECT column_name 
FROM table_name
WHERE column = ANY (SELECT column_name FROM table_name WHERE condition);

SELECT column_name 
FROM table_name
WHERE column = ALL (SELECT column_name FROM table_name WHERE condition);

-- WHERE NOT
SELECT column_name 
FROM table_name
WHERE NOT condition;

-- CASE Statement
SELECT 
    CASE
        WHEN condition THEN 'true'
        ELSE 'false'
    END AS Result
FROM table_name;

-- INSERT INTO Table
-- Specify columns
INSERT INTO table_name (column, column)
VALUES (value, value);

-- Insert to all columns
INSERT INTO table_name
VALUES (value, value);

-- UPDATE Table
UPDATE table_name
SET column1 = value1,
    column2 = value2,
    column3 = value3
WHERE condition;

-- DELETE Table
DELETE FROM table_name WHERE condition;

-- TRUNCATE Table
TRUNCATE TABLE table_name;

-- BASIC FUNCTIONS
-- SELECT TOP
SELECT TOP (number | percent) column_name
FROM table_name
WHERE condition;

-- MIN/MAX
SELECT MIN(column_name) AS MinValue,
       MAX(column_name) AS MaxValue
FROM table_name
WHERE condition;

-- COUNT/AVG/SUM
SELECT COUNT(column_name) AS TotalCount,
       AVG(column_name) AS AverageValue,
       SUM(column_name) AS SumValue
FROM table_name
WHERE condition;

-- LIKE Syntax
SELECT column
FROM table_name
WHERE column LIKE pattern;

-- Pattern Operators
WHERE column LIKE 'a%'  -- Finds values starting with "a"
WHERE column LIKE '%a'  -- Finds values ending with "a"
WHERE column LIKE '%or%' -- Finds values containing "or"
WHERE column LIKE '_r%' -- Finds values with "r" in the second position

-- STUFF function
SELECT STUFF(column_name, start, length, new_string) 
FROM table_name;

-- REPLACE function
SELECT REPLACE(string, old_string, new_string);

-- COALESCE function
SELECT COALESCE(NULL, NULL, NULL, 'DefaultValue', NULL);

-- JOINS
-- Different types of JOINS:
-- INNER JOIN: Returns matching records in both tables.
-- LEFT JOIN: Returns all records from the left table, matched records from the right table.
-- RIGHT JOIN: Returns all records from the right table, matched records from the left table.
-- FULL JOIN: Returns records with a match in either table.

SELECT A.column_name, B.column_name
FROM table_A A
JOIN table_B B ON A.column = B.column;

-- SQL STATEMENTS
-- CREATE Table
CREATE TABLE table_name (
    ID int IDENTITY(1,1) PRIMARY KEY,
    column1 data_type NOT NULL,
    column2 data_type NOT NULL FOREIGN KEY REFERENCES other_table(column)
);

-- ALTER Table
ALTER TABLE table_name
ADD column_name data_type;

ALTER TABLE table_name
DROP COLUMN column_name;

ALTER TABLE table_name
ALTER COLUMN column_name new_data_type;

-- CHECK Constraint
CREATE TABLE table_name (
    ID int NOT NULL,
    percentage DECIMAL(5,4),
    CHECK (percentage <= 1.0000)
);

-- DEFAULT Constraint
CREATE TABLE table_name (
    ID int NOT NULL,
    text_column varchar(255) DEFAULT 'DefaultText',
    int_column INT DEFAULT 1
);

-- IF ELSE Statement
IF (@variable = 1)
BEGIN
    -- insert code
END
ELSE
BEGIN
    -- insert code
END;

-- TEMPORARY Table
DECLARE @Temp TABLE (column1 INT, column2 VARCHAR(10));

-- STORED PROCEDURES
-- Template for GET procedure
CREATE PROCEDURE [dbo].[SP_Template_Get]
    @Parameter INT,
    @PageNumber INT = 1,
    @PageSize INT = 20
AS
BEGIN
    -- logic here
END;

-- Template for SET procedure
CREATE PROCEDURE [dbo].[SP_Template_Set]
    @Parameter INT
AS
BEGIN
    BEGIN TRY
        BEGIN TRANSACTION;
        -- logic here
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
    END CATCH;
END;

-- CTE Example
WITH CTE_Example AS (
    SELECT column1, column2
    FROM table_name
    WHERE condition
)
SELECT *
FROM CTE_Example;

-- OFFSET FETCH Example
SELECT column1, column2 
FROM table_name 
ORDER BY column1 
OFFSET 10 ROWS FETCH NEXT 5 ROWS ONLY;

-- TRY-CATCH Statement
BEGIN TRY
    BEGIN TRANSACTION;
    -- code to execute
    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    ROLLBACK TRANSACTION;
    DECLARE @ErrorMessage NVARCHAR(4000), @ErrorSeverity INT, @ErrorState INT;
    SELECT @ErrorMessage = ERROR_MESSAGE(), @ErrorSeverity = ERROR_SEVERITY(), @ErrorState = ERROR_STATE();
    RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
END CATCH;

-- CURSOR Example
DECLARE cursor_name CURSOR FOR 
SELECT column1, column2 FROM table_name;

OPEN cursor_name;
FETCH NEXT FROM cursor_name INTO @var1, @var2;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- logic here
    FETCH NEXT FROM cursor_name INTO @var1, @var2;
END;

CLOSE cursor_name;
DEALLOCATE cursor_name;

-- FINAL NOTES
-- Other handy commands and tricks have been included in this comprehensive cheat sheet. For more advanced queries, explore official SQL Server documentation.



-- ADDITIONAL SQL CHEATSHEET
-- New advanced commands and concepts for further SQL understanding

-- CONCATENATION of Strings
-- Combine multiple strings into one.
SELECT CONCAT(first_name, ' ', last_name) AS full_name 
FROM employees;

-- DISTINCT Keyword with COUNT
-- Count distinct occurrences of a specific column.
SELECT COUNT(DISTINCT column_name) AS DistinctCount 
FROM table_name;

-- INNER JOIN with Multiple Conditions
-- Join tables with multiple conditions.
SELECT A.column1, B.column2 
FROM table_A A 
INNER JOIN table_B B 
    ON A.id = B.id AND A.status = B.status;

-- CROSS JOIN Example
-- Combines each row of the first table with all rows of the second table.
SELECT A.column, B.column 
FROM table_A A 
CROSS JOIN table_B B;

-- SELF JOIN Example
-- Join a table with itself to find relationships within the same table.
SELECT A.employee_name, B.manager_name 
FROM employees A, employees B 
WHERE A.manager_id = B.employee_id;

-- INDEXED VIEW Example
-- Create a view with an index for faster retrieval.
CREATE VIEW view_name WITH SCHEMABINDING AS 
SELECT column1, column2 
FROM schema.table_name;
GO

CREATE UNIQUE CLUSTERED INDEX idx_view ON view_name(column1);

-- STORED PROCEDURE with OUTPUT Parameter
-- Stored procedure with an output parameter to return values.
CREATE PROCEDURE GetEmployeeCount 
    @DepartmentId INT,
    @EmployeeCount INT OUTPUT
AS
BEGIN
    SELECT @EmployeeCount = COUNT(*)
    FROM employees
    WHERE department_id = @DepartmentId;
END;

-- EXEC GetEmployeeCount @DepartmentId = 1, @EmployeeCount = @OutputVar OUTPUT;

-- DYNAMIC SQL Example
-- Execute dynamically generated SQL statements using sp_executesql.
DECLARE @SQL NVARCHAR(MAX);
SET @SQL = 'SELECT * FROM ' + @TableName;
EXEC sp_executesql @SQL;

-- SUBQUERIES
-- Using subqueries in SELECT and WHERE clauses.
SELECT employee_id, (SELECT MAX(salary) FROM employees) AS MaxSalary 
FROM employees;

-- CONDITIONAL AGGREGATION
-- Aggregate data conditionally using CASE in aggregate functions.
SELECT 
    department,
    COUNT(CASE WHEN status = 'Active' THEN 1 END) AS ActiveCount,
    COUNT(CASE WHEN status = 'Inactive' THEN 1 END) AS InactiveCount
FROM employees
GROUP BY department;

-- RANKING FUNCTIONS
-- Use RANK(), DENSE_RANK(), ROW_NUMBER() to rank rows based on certain conditions.
SELECT 
    column_name,
    RANK() OVER (ORDER BY column_name DESC) AS Rank
FROM table_name;

-- PIVOT TABLE
-- Convert rows into columns using PIVOT.
SELECT *
FROM (
    SELECT column_name, row_value
    FROM table_name
) AS SourceTable
PIVOT (
    SUM(row_value) 
    FOR column_name IN ([Column1], [Column2], [Column3])
) AS PivotTable;

-- UNPIVOT Example
-- Convert columns into rows using UNPIVOT.
SELECT 
    unpivoted_column, unpivoted_value
FROM (
    SELECT column1, column2, column3
    FROM table_name
) AS SourceTable
UNPIVOT (
    unpivoted_value FOR unpivoted_column IN (column1, column2, column3)
) AS UnpivotTable;

-- WINDOW FUNCTIONS
-- Use functions like LAG(), LEAD(), FIRST_VALUE(), and LAST_VALUE() over partitions.
SELECT 
    column1,
    LAG(column1, 1, 0) OVER (ORDER BY column1) AS PrevValue
FROM table_name;

-- COMMON SYSTEM FUNCTIONS
-- GETDATE(): Returns the current system date and time.
SELECT GETDATE() AS CurrentDateTime;

-- SYSTEM_USER: Returns the name of the current user.
SELECT SYSTEM_USER AS CurrentUser;

-- TRY_PARSE/TRY_CONVERT
-- Safely convert data types and return NULL if conversion fails.
SELECT TRY_PARSE('2021-10-01' AS DATE) AS ParsedDate;

-- JSON FUNCTIONS
-- Parse JSON data using OPENJSON.
SELECT * 
FROM OPENJSON('{
    "id": 1,
    "name": "John",
    "department": "Sales"
}')
WITH (
    id INT,
    name NVARCHAR(50),
    department NVARCHAR(50)
);

-- XML FUNCTIONS
-- Parse XML data using nodes and extract values using the .value() method.
DECLARE @xml XML = '<employees><employee id="1"><name>John</name></employee></employees>';
SELECT 
    emp.value('@id', 'INT') AS EmployeeID,
    emp.value('(name)[1]', 'NVARCHAR(50)') AS Name
FROM @xml.nodes('/employees/employee') AS e(emp);

-- TEMPORAL TABLES
-- Track historical data changes using system versioning.
CREATE TABLE TemporalTable (
    ID INT PRIMARY KEY,
    Name NVARCHAR(100),
    SysStartTime DATETIME2 GENERATED ALWAYS AS ROW START,
    SysEndTime DATETIME2 GENERATED ALWAYS AS ROW END,
    PERIOD FOR SYSTEM_TIME (SysStartTime, SysEndTime)
)
WITH (SYSTEM_VERSIONING = ON (HISTORY_TABLE = dbo.TemporalTableHistory));

-- CREATING A SEQUENCE
-- Create a sequence object to auto-generate sequential numbers.
CREATE SEQUENCE seq_example
START WITH 1
INCREMENT BY 1;

-- Use the sequence to insert values.
INSERT INTO table_name (column1)
VALUES (NEXT VALUE FOR seq_example);

-- SQL MERGE Statement
-- Combine INSERT, UPDATE, and DELETE operations into a single statement.
MERGE INTO target_table AS target
USING source_table AS source
ON target.id = source.id
WHEN MATCHED THEN
    UPDATE SET target.column = source.column
WHEN NOT MATCHED BY TARGET THEN
    INSERT (column) VALUES (source.column)
WHEN NOT MATCHED BY SOURCE THEN
    DELETE;

-- TRANSACTION MANAGEMENT
-- Explicitly begin, commit, or roll back transactions.
BEGIN TRANSACTION;
    -- SQL statements
COMMIT TRANSACTION;
-- or
ROLLBACK TRANSACTION;

-- SQL SCRIPT FOR PERMISSIONS
-- Grant read permission to a user on a specific table.
GRANT SELECT ON table_name TO user_name;

-- REVOKE permission from a user.
REVOKE INSERT ON table_name FROM user_name;

-- DENY permission to a user.
DENY DELETE ON table_name TO user_name;



-- create database objects and constructs
CREATE TABLE -- (creates a new table)
CREATE VIEW -- (creates a virtual table based on a query)
CREATE INDEX -- (creates an index for faster data retrieval)
CREATE PROCEDURE -- (creates a stored procedure for reusable SQL code)
CREATE FUNCTION -- (creates a user-defined function that returns a value or table)
CREATE TRIGGER -- (creates a trigger to automatically execute code based on an event)
CREATE SCHEMA -- (creates a new schema for logical grouping of objects)
CREATE SEQUENCE -- (creates a sequence to generate unique numeric values)
CREATE TYPE -- (creates a custom data type)
CREATE SYNONYM -- (creates an alias for a database object)
CREATE CONSTRAINT -- (creates a rule on a column or table, e.g., CHECK, UNIQUE)
CREATE CURSOR -- (creates a cursor for row-by-row processing in SQL code)
CREATE TEMPORARY TABLE -- (creates a table that exists only for the duration of a session)
## Product Metrics (15 questions)

#### 1. What would be good metrics of success for an advertising-driven consumer product? (Buzzfeed, YouTube, Google Search, etc.) A service-driven consumer product? (Uber, Flickr, Venmo, etc.)
  * advertising-driven: Pageviews and daily actives, CTR, CPC (cost per click)
    * click-ads  
    * display-ads  
  * service-driven: number of purchases, conversion rate
#### 2. What would be good metrics of success for a productiv- ity tool? (Evernote, Asana, Google Docs, etc.) A MOOC? (edX, Coursera, Udacity, etc.)
  * productivity tool: same as premium subscriptions
  * MOOC: same as premium subscriptions, completion rate
#### 3. What would be good metrics of success for an e-commerce product? (Etsy, Groupon, Birchbox, etc.) A subscrip- tion product? (Net ix, Birchbox, Hulu, etc.) Premium subscriptions? (OKCupid, LinkedIn, Spotify, etc.) 
  * e-commerce: number of purchases, conversion rate, Hourly, daily, weekly, monthly, quarterly, and annual sales, Cost of goods sold, Inventory levels, Site traffic, Unique visitors versus returning visitors, Customer service phone call count, Average resolution time
  * subscription
    * churn, CoCA, ARPU, MRR, LTV
  * premium subscriptions: 

#### 4. What would be good metrics of success for a consumer product that relies heavily on engagement and interac- tion? (Snapchat, Pinterest, Facebook, etc.) A messaging product? (GroupMe, Hangouts, Snapchat, etc.)
  * heavily on engagement and interaction: uses AU ratios, email summary by type, and push notification summary by type, resurrection ratio
  * messaging product: 
#### 5. What would be good metrics of success for a product that o ered in-app purchases? (Zynga, Angry Birds, other gaming apps)
  * Average Revenue Per Paid User
  * Average Revenue Per User
#### 6. A certain metric is violating your expectations by going down or up more than you expect. How would you try to identify the cause of the change?
  * breakdown the KPI’s into what consists them and find where the change is
  * then further breakdown that basic KPI by channel, user cluster, etc. and relate them with any campaigns, changes in user behaviors in that segment
#### 7. Growth for total number of tweets sent has been slow this month. What data would you look at to determine the cause of the problem?
  * look at competitors' tweet growth
  * look at your social media engagement on other platforms
  * look at your sales data 
#### 8. You’re a restaurant and are approached by Groupon to run a deal. What data would you ask from them in order to determine whether or not to do the deal?
  * for similar restaurants (they should define similarity), average increase in revenue gain per coupon, average increase in customers per coupon, number of meals sold
#### 9. You are tasked with improving the e ciency of a subway system. Where would you start?
  * define efficiency
#### 10. Say you are working on Facebook News Feed. What would be some metrics that you think are important? How would you make the news each person gets more relevant?
  * rate for each action, duration users stay, CTR for sponsor feed posts
  * ref. News Feed Optimization
    * Affinity score: how close the content creator and the users are
    * Weight: weight for the edge type (comment, like, tag, etc.). Emphasis on features the company wants to promote
    * Time decay: the older the less important
#### 11. How would you measure the impact that sponsored stories on Facebook News Feed have on user engagement? How would you determine the optimum balance between sponsored stories and organic content on a user’s News Feed?
  * AB test on different balance ratio and see 
#### 12. You are on the data science team at Uber and you are asked to start thinking about surge pricing. What would be the objectives of such a product and how would you start looking into this?
  *  there is a gradual step-function type scaling mechanism until that imbalance of requests-to-drivers is alleviated and then vice versa as too many drivers come online enticed by the surge pricing structure. 
  * I would bet the algorithm is custom tailored and calibrated to each location as price elasticities almost certainly vary across different cities depending on a huge multitude of variables: income, distance/sprawl, traffic patterns, car ownership, etc. With the massive troves of user data that Uber probably has collected, they most likely have tweaked the algos for each city to adjust for these varying sensitivities to surge pricing. Throw in some machine learning and incredibly rich data and you've got yourself an incredible, constantly-evolving algorithm.  

#### 13. Say that you are Netflix. How would you determine what original series you should invest in and create?
  * Netflix uses data to estimate the potential market size for an original series before giving it the go-ahead.
#### 14. What kind of services would  nd churn (metric that tracks how many customers leave the service) helpful? How would you calculate churn?
  * subscription based services
#### 15. Let’s say that you’re are scheduling content for a content provider on television. How would you determine the best times to schedule content?




# Product Metrics Guide

## Advertising-Driven Consumer Products
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Pageviews | Total number of pages viewed by users | Sum of all page loads | Measures content consumption | Buzzfeed, YouTube |
| Daily Active Users (DAU) | Number of unique users who engage with product daily | Count of unique user IDs with at least one session in a 24-hour period | Measures user engagement | Google Search, YouTube |
| Click-Through Rate (CTR) | Percentage of impressions that result in clicks | (Number of Clicks / Number of Impressions) × 100% | Measures ad effectiveness | Google Search, Facebook |
| Cost Per Click (CPC) | Average cost paid per ad click | Total Ad Spend / Number of Clicks | Measures ad monetization | Google Ads |
| Cost Per Mille (CPM) | Cost per thousand impressions | (Total Ad Spend / Number of Impressions) × 1000 | Alternative ad pricing model | Display advertising |
| Impression-to-Conversion | Ratio of ad views to conversions | Number of Conversions / Number of Impressions | Measures ad effectiveness | Banner ads |
| Return on Ad Spend (ROAS) | Revenue generated per dollar spent on ads | Revenue Generated from Ads / Cost of Ads | Measures ad campaign ROI | All ad platforms |
| Ad Viewability | Percentage of ads that are actually viewable | (Number of Viewable Impressions / Total Impressions) × 100% | Measures ad quality | Display advertising |

## Service-Driven Consumer Products
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Number of Purchases | Total completed transactions | Count of all completed transactions | Measures service usage | Uber, Venmo |
| Conversion Rate | Percentage of users who complete desired action | (Number of Conversions / Number of Total Users) × 100% | Measures service effectiveness | Uber, Flickr |
| Customer Acquisition Cost (CAC) | Cost to acquire a new customer | Total Marketing and Sales Costs / Number of New Customers | Measures marketing efficiency | Uber, Venmo |
| Average Order Value (AOV) | Average amount spent per transaction | Total Revenue / Number of Orders | Measures customer spending | Uber, food delivery |
| Customer Lifetime Value (LTV) | Predicted revenue from a customer | (Average Revenue per Customer per Period × Average Customer Lifespan) - CAC | Measures long-term customer value | Uber, Flickr |
| Net Promoter Score (NPS) | Likelihood of customer recommending service | % Promoters (9-10 scores) - % Detractors (0-6 scores) | Measures customer satisfaction | All service products |
| Time to Service | Time between request and delivery | Average(Service Delivery Time - Service Request Time) | Measures service efficiency | Uber, delivery services |
| Repeat Usage Rate | Percentage of users who return | (Number of Repeat Users / Total Users) × 100% | Measures service stickiness | Venmo, Uber |

## Productivity Tools
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Daily/Weekly Active Users | Number of unique active users | Count of unique users in time period | Measures consistent usage | Evernote, Asana |
| Task Completion Rate | Percentage of started tasks completed | (Number of Completed Tasks / Number of Created Tasks) × 100% | Measures tool effectiveness | Asana, Jira |
| Documents Created/Edited | Number of new/modified documents | Count of document creation/modification events | Measures content creation | Google Docs |
| Feature Adoption | Percentage of users using specific features | (Users Using Feature / Total Active Users) × 100% | Measures feature value | Evernote, Asana |
| Time Saved | Estimated time saved by using tool | Traditional Process Time - Tool Process Time | Measures productivity gain | All productivity tools |
| Collaboration Rate | Number of shared/collaborative projects | Count of projects with multiple contributors / Total Projects | Measures team usage | Google Docs, Asana |
| Integration Usage | Use of integrations with other tools | (Number of Users Using Integrations / Total Users) × 100% | Measures ecosystem value | Slack, Asana |
| Premium Conversion | Free to paid conversion rate | (Number of Free Users Converting to Paid / Total Free Users) × 100% | Measures monetization | Evernote, Asana |

## MOOCs (Online Education)
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Course Completion Rate | Percentage of students finishing courses | (Number of Students Completing Course / Number of Enrolled Students) × 100% | Measures content quality | Coursera, edX |
| Premium Subscriptions | Number of paid subscriptions | Count of active paid subscriptions | Measures monetization | Udacity, Coursera |
| Quiz/Assignment Scores | Average performance on assessments | Sum of All Scores / Number of Submissions | Measures learning effectiveness | All MOOCs |
| Video Engagement | Watch time and completion rate | (Total Watch Time / (Video Length × Number of Viewers)) × 100% | Measures content quality | All MOOCs |
| Forum Participation | Number of discussions and replies | Count of forum posts and replies per user | Measures community engagement | edX, Coursera |
| Certificate Purchases | Number of certificates issued | Count of paid certificates issued | Measures perceived value | Coursera, edX |
| Student-to-Student Interactions | Collaborative activities | Average peer reviews or comments per student | Measures peer learning | Group projects |
| Course Rating | Student satisfaction scores | Average of all course ratings (typically 1-5 scale) | Measures perceived quality | All MOOCs |

## E-commerce Products
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Number of Purchases | Total completed transactions | Count of all completed orders | Measures sales volume | Etsy, Groupon |
| Conversion Rate | Percentage of visitors who purchase | (Number of Transactions / Number of Visitors) × 100% | Measures site effectiveness | All e-commerce |
| Average Order Value (AOV) | Average amount spent per order | Total Revenue / Number of Orders | Measures customer spending | Etsy, Amazon |
| Shopping Cart Abandonment | Percentage of abandoned carts | (Number of Abandoned Carts / Number of Created Carts) × 100% | Identifies checkout issues | All e-commerce |
| Cost of Goods Sold | Direct costs of products sold | Sum of product costs, shipping, packaging, etc. | Measures product profitability | All e-commerce |
| Inventory Turnover | Rate at which inventory is sold | Cost of Goods Sold / Average Inventory Value | Measures supply chain efficiency | All e-commerce |
| Return Rate | Percentage of purchases returned | (Number of Returns / Number of Sales) × 100% | Measures product satisfaction | All e-commerce |
| Customer Acquisition Cost | Cost to acquire new customers | Marketing Expenses / Number of New Customers | Measures marketing efficiency | All e-commerce |

## Subscription Products
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Churn Rate | Percentage of subscribers who cancel | (Customers Lost in Period / Customers at Start of Period) × 100% | Measures retention | Netflix, Hulu |
| Cost of Customer Acquisition (CoCA) | Cost to acquire new subscribers | Total Acquisition Costs / Number of New Customers | Measures marketing efficiency | All subscription services |
| Average Revenue Per User (ARPU) | Average revenue generated per user | Total Revenue / Total Number of Users | Measures monetization | Spotify, Netflix |
| Monthly Recurring Revenue (MRR) | Predictable monthly revenue | Sum of all monthly subscription fees | Measures business stability | All subscription services |
| Customer Lifetime Value (LTV) | Total value of a customer relationship | ARPU × Average Customer Lifespan | Measures long-term value | Netflix, Spotify |
| Renewal Rate | Percentage of subscribers who renew | (Number of Renewals / Number of Subscriptions Due for Renewal) × 100% | Measures satisfaction | Annual subscriptions |
| Free-to-Paid Conversion | Free trial to paid conversion rate | (Number of Free Trials Converting to Paid / Total Free Trials) × 100% | Measures acquisition funnel | Spotify, LinkedIn Premium |
| Net Revenue Retention | Revenue retention including expansions | (MRR at Period End - New MRR) / MRR at Period Start | Measures growth from existing customers | Tiered subscriptions |

## Engagement-Driven Products
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Daily/Weekly/Monthly Active Users Ratio | Relationship between activity periods | DAU/WAU or WAU/MAU | Measures engagement consistency | Facebook, Snapchat |
| Resurrection Ratio | Returned users after inactivity | Number of Returning Inactive Users / Total Previously Inactive Users | Measures re-engagement | Pinterest, Facebook |
| Email Engagement | Open/click rates by type | (Number of Opens or Clicks / Number of Emails Sent) × 100% | Measures notification effectiveness | All platforms |
| Push Notification Engagement | Open rates by type | (Number of Opens / Number of Notifications Sent) × 100% | Measures mobile engagement | All mobile apps |
| Session Duration | Time spent per visit | Sum of All Session Times / Number of Sessions | Measures content appeal | Facebook, Pinterest |
| Content Creation Rate | User-generated content volume | Pieces of Content Created / Number of Active Users | Measures contribution | Instagram, YouTube |
| Viral Coefficient | New users generated by existing users | Average Invites Sent × Conversion Rate of Invites | Measures organic growth | Facebook, Pinterest |
| Stickiness | DAU/MAU ratio | DAU / MAU × 100% | Measures habit formation | All engagement platforms |

## Messaging Products
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Messages Sent Per User | Average messaging volume | Total Messages Sent / Number of Active Users | Measures activity level | GroupMe, Snapchat |
| Response Rate | Percentage of messages receiving replies | (Number of Messages with Replies / Total Messages Sent) × 100% | Measures conversation depth | All messaging apps |
| Group Creation | Number of new groups formed | Count of new groups created in period | Measures network growth | GroupMe, WhatsApp |
| Media Sharing Rate | Photos/videos shared per user | Number of Media Items Shared / Number of Active Users | Measures rich content use | Snapchat, WhatsApp |
| Time to Response | Average reply time | Average(Reply Timestamp - Message Timestamp) | Measures conversation flow | All messaging apps |
| Contacts Added | New connections per user | Number of New Contacts Added / Number of Users | Measures network growth | All messaging apps |
| Retention by Cohort | User retention over time | (Users Active in Current Period from Cohort / Original Cohort Size) × 100% | Measures long-term value | All messaging apps |
| Cross-Platform Usage | Activity across devices | Percentage of Users Active on Multiple Platforms | Measures platform flexibility | Hangouts, WhatsApp |

## In-App Purchase Products
| Metric | Description | Calculation | Application | Example Products |
|--------|-------------|-------------|-------------|-----------------|
| Average Revenue Per Paid User (ARPPU) | Revenue per monetized user | Total Revenue / Number of Paying Users | Measures paying user value | Zynga, Angry Birds |
| Average Revenue Per User (ARPU) | Revenue across all users | Total Revenue / Total Number of Users | Measures overall monetization | All gaming apps |
| Conversion to Paid | Percentage of users who make purchases | (Number of Paying Users / Total Users) × 100% | Measures monetization success | Free-to-play games |
| Purchase Frequency | Purchases per paying user | Number of Purchases / Number of Paying Users | Measures repeat monetization | All gaming apps |
| Average Transaction Value | Average spend per purchase | Total Revenue / Number of Transactions | Measures purchase optimization | All gaming apps |
| Item Purchase Distribution | Popularity of different items | (Purchases of Specific Item / Total Purchases) × 100% | Guides in-game economy | All gaming apps |
| First Purchase Time | Time from install to first purchase | Average(First Purchase Time - Install Time) | Measures conversion speed | All gaming apps |
| Paying User Concentration | Revenue % from top spenders | (Revenue from Top X% of Spenders / Total Revenue) × 100% | Measures revenue distribution | All gaming apps |

## A/B Testing Metrics
| Metric | Description | Calculation | Application | Testing Focus |
|--------|-------------|-------------|-------------|--------------|
| Statistical Significance | Confidence level in results | p-value < significance level (typically 0.05) | Validates test reliability | All A/B tests |
| Conversion Lift | Percentage improvement in conversions | ((Test Conversion Rate - Control Conversion Rate) / Control Conversion Rate) × 100% | Measures test impact | Conversion optimization |
| Revenue Per User Lift | Increase in average user value | ((Test ARPU - Control ARPU) / Control ARPU) × 100% | Measures monetization impact | Pricing tests |
| Engagement Lift | Increase in user activity | ((Test Engagement - Control Engagement) / Control Engagement) × 100% | Measures feature impact | UI/UX changes |
| Sample Size Adequacy | Sufficient data for conclusions | Calculate using power analysis based on minimum detectable effect | Ensures test validity | All A/B tests |
| Time to Significance | Days required for reliable results | Time taken to reach statistically significant results | Measures test efficiency | All A/B tests |
| Segmented Impact | Test effects on different user groups | Compare lift metrics across different user segments | Identifies varying responses | Personalization tests |
| Return on Investment | Financial return on test implementation | (Incremental Revenue - Cost of Implementation) / Cost of Implementation | Measures business impact | Major feature changes |

## Troubleshooting Metrics
| Metric | Description | Calculation | Application | Analysis Approach |
|--------|-------------|-------------|-------------|------------------|
| KPI Breakdown | Analyzing components of key metrics | Decompose metric into constituent parts and analyze each | Identifies specific issues | Component analysis |
| Segment Performance | Metrics by channel, cluster, etc. | Calculate primary metrics for each segment separately | Isolates problem areas | Segmentation analysis |
| Anomaly Detection | Identification of unusual patterns | (Current Value - Historical Average) / Standard Deviation | Flags outliers | Statistical analysis |
| Correlation Analysis | Relationship between metrics | Pearson or Spearman correlation coefficient | Connects related metrics | Statistical analysis |
| Behavioral Cohort Analysis | Metrics by user acquisition time | Calculate metrics separately for each cohort | Identifies evolving usage | Cohort analysis |
| Funnel Conversion | Step-by-step conversion rates | (Number at Current Step / Number at Previous Step) × 100% | Finds conversion barriers | Funnel analysis |
| Error Rate | Frequency of technical failures | (Number of Errors / Number of Transactions) × 100% | Identifies technical issues | Performance monitoring |
| Attribution Analysis | Source of user actions | User Actions Attributed to Source / Total User Actions | Maps cause and effect | Marketing analysis |





## Comprehensive Evaluation Metrics for LLM Chatbots with RAG and Agents

### Retrieval and Context Metrics
| Metric | Description | Calculation Method | Application | Interpretation |
|--------|-------------|-------------------|-------------|----------------|
| Retrieval Precision@k | Percentage of top-k retrieved documents that are relevant | (Number of Relevant Documents in Top-k / k) × 100% | RAG quality assessment | Higher precision indicates better document selection quality |
| Recall@k | Percentage of all relevant documents that appear in top-k results | (Number of Relevant Documents in Top-k / Total Relevant Documents) × 100% | RAG coverage assessment | Higher recall indicates better retrieval of available information |
| Context Relevance | Degree to which retrieved context contains information needed to answer query | Human evaluation on 1-5 scale or relevance score between query and retrieved passages | RAG pre-processing quality | Higher scores indicate better context selection |
| Context Utilization | Effectiveness of incorporating retrieved information into responses | Semantic overlap between key information in context and response; can use metrics like ROUGE | RAG integration efficiency | Higher scores indicate better utilization of retrieved information |
| Citation Accuracy | Correctness of information attribution to sources | (Number of Correct Citations / Total Citations) × 100% | RAG system trustworthiness | Higher accuracy indicates reliable source attribution |

### LLM Response Quality Metrics
| Metric | Description | Calculation Method | Application | Interpretation |
|--------|-------------|-------------------|-------------|----------------|
| Response Relevance | Measures how well the response addresses the user's query | Human evaluation on 1-5 scale or automated semantic similarity scores | Core response quality assessment | Higher scores indicate better alignment with user intent |
| Faithfulness/Hallucination Rate | Measures the factual accuracy of generated content | Percentage of claims in response that are verifiable; (Verified Claims / Total Claims) × 100% | Content reliability | Lower hallucination rates indicate higher factual accuracy |
| Coherence | Logical flow and consistency within the response | Human evaluation or automated coherence scoring (e.g., inter-sentence consistency metrics) | Response readability | Higher scores indicate better logical structure |
| Helpfulness | Degree to which the response actually solves the user's problem | Human evaluation or task completion verification | User value assessment | Higher scores indicate more actionable or valuable responses |
| Conciseness | Appropriate length and information density of response | Information content relative to word count or semantic density scoring | Response efficiency | Optimal scores balance completeness with brevity |

### Agent Performance Metrics
| Metric | Description | Calculation Method | Application | Interpretation |
|--------|-------------|-------------------|-------------|----------------|
| Task Completion Rate | Percentage of tasks successfully completed by agents | (Successfully Completed Tasks / Total Tasks) × 100% | Overall agent effectiveness | Higher rates indicate better overall system utility |
| Tool Selection Accuracy | Correctness of agent's tool selection for specific tasks | (Number of Appropriate Tool Selections / Total Tool Uses) × 100% | Agent reasoning quality | Higher accuracy indicates better reasoning about tool applicability |
| Agent Planning Quality | Effectiveness of agent's planning and execution strategy | Human evaluation or automated assessment of plan validity/optimality | Agent reasoning path | Higher scores indicate more efficient problem-solving approach |
| Agent Adaptability | Ability to adjust strategy when encountering unexpected situations | Success rate in handling edge cases or recovery from initial failures | Agent robustness | Higher adaptability indicates more resilient agent systems |
| Action Efficiency | Minimization of unnecessary actions in completing a task | (Minimum Possible Actions / Actions Taken) × 100% | Agent optimization | Higher efficiency indicates more direct path to task completion |

### System Performance Metrics
| Metric | Description | Calculation Method | Application | Interpretation |
|--------|-------------|-------------------|-------------|----------------|
| End-to-End Latency | Total time from query submission to complete response delivery | Average response time in seconds across queries | System performance | Lower latency indicates better real-time capabilities |
| Token Efficiency | Information value relative to token usage | Information density: semantic content value divided by token count | System cost optimization | Higher efficiency indicates better value for computational resources |
| User Satisfaction | Direct measurement of user contentment with responses | Average rating on 1-5 scale from explicit user feedback | Ultimate success metric | Higher scores indicate better meeting of user needs |
| Conversation Success Rate | Percentage of conversations where user goals were achieved | (Number of Successful Conversations / Total Conversations) × 100% | Holistic evaluation | Higher rates indicate better overall system effectiveness |
| Error Recovery Rate | System's ability to recover from misunderstandings | (Number of Successful Recoveries / Total Errors) × 100% | System robustness | Higher rates indicate more resilient conversation handling |
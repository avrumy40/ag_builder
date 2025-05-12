# AG Hierarchy Builder User Guide

## What Problem Does This Solve?

The AG Hierarchy Builder is a sophisticated tool for retail professionals who need to create and analyze **Assortment Groups (AGs)** - logical clusters of similar products based on shared attributes. This tool solves several key challenges in retail product management:

1. **Product Classification**: Intelligently groups products based on similar attributes to create coherent assortment groups.
2. **Hierarchy Building**: Creates multi-level product hierarchies for better organization and navigation.
3. **Assortment Analysis**: Provides visual insights about product distribution across groups to ensure balanced assortments.
4. **Attribute Selection**: Helps determine which product attributes create the most effective groupings.
5. **Price Variance Monitoring**: Identifies and analyzes price inconsistencies within product groups.

## How to Use the AG Hierarchy Builder

### Step 1: Upload Your Product Catalog

1. Prepare a CSV file containing your product data. Required columns include:
   - `product_id` (unique identifier for each product)
   - Any attribute columns you want to use for grouping (brand, color, size, etc.)
   - `price` (optional, enables price variance analysis)

2. Click the **"Browse files"** button to upload your CSV file.

3. The system will show a preview of your uploaded data. Verify it looks correct before proceeding.

### Step 2: Analyze Product Attributes

1. The application automatically analyzes all attributes (columns) in your data.

2. View the distribution of each attribute, sorted by number of unique values:
   - **Categorical attributes**: Shows the number of unique values and their distribution
   - **Numeric attributes**: Displays histograms of value distributions

3. Use this analysis to understand which attributes might be good candidates for grouping products.

### Step 3: Select Attributes for AG Creation

1. **Manual Selection Tab**:
   - Add combinations of attributes to test by clicking the "+ Add Combination" button
   - Select attributes from the dropdown menus for each combination
   - Remove combinations by clicking the "×" icon

2. **Recommended Combinations Tab**:
   - Set filtering criteria:
     - Min/Max products per AG
     - Min/Max number of AGs
     - Exclude attributes you don't want to consider
   - Click "Generate Recommendations" to see suggested attribute combinations
   - Each recommendation shows a score from 1-3 (higher is better) with explanation
   - Click the graph icon (📊) on any recommendation card to view its detailed results

3. Click "Compare Selected Combinations" to move to the results page with your chosen combinations.

### Step 4: View Results

1. Navigate between your selected combinations using the tabs at the top.

2. For each combination, the system displays:
   - **AG Statistics**: Total products, number of AGs, average and median products per AG
   - **Products per AG Distribution**: Histogram showing how many AGs have a certain product count
   - **Top AGs by Product Count**: Bar chart of the largest AGs
   - **Price Variation**: Analysis of price consistency within each AG (if price data is available)
   - **Hierarchy Levels** (if enabled): Analysis of each level in the attribute hierarchy

3. Hover over any chart element to see detailed information in tooltips.

4. Based on these visualizations, select the attribute combination that creates the most balanced and logical product groupings for your business needs.

## Key Features

- **Interactive Visualizations**: Hover over charts to see detailed information with tooltips
- **Scoring System**: Evaluates attribute combinations on a 1-3 scale based on distribution normality and price variance
- **Hierarchy Mode**: Creates nested levels of AGs for more detailed organization
- **Intelligent Recommendations**: Suggests optimal attribute combinations based on your data
- **Comparison View**: Enables side-by-side analysis of different attribute combinations

## Tips for Best Results

1. **Clean Your Data**: Remove duplicates and fill missing values before uploading.
2. **Start with Fewer Attributes**: Begin with 2-3 attributes and add more if needed.
3. **Balance Distribution**: Look for combinations that create a relatively even distribution of products across AGs.
4. **Consider Price Variance**: Lower price variance within AGs often indicates more coherent groupings.
5. **Use the Scoring System**: Pay attention to the automated scoring to quickly identify promising combinations.

## Technical Notes

- The application saves generated AG combinations to CSV files in the `output` directory.
- Large datasets may take longer to process during the recommendation generation phase.
- For optimal performance, consider limiting the number of attribute combinations you compare simultaneously.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file,sep="\t")
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return len(self.chipo)
    
    def info(self) -> None:
        # TODO
        # print data info.
        pass
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(list(self.chipo.columns))
    
    def most_ordered_item(self):
        # TODO
        data = self.chipo.groupby(['item_name'])['quantity'].sum().reset_index().nlargest(1,"quantity")
        item_name = data['item_name'].item()
        quantity = data['quantity'].item()
        return item_name, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       return self.chipo['quantity'].sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        self.chipo['item_price'] = self.chipo.apply(lambda row:float(row['item_price'].split('$')[1].strip()), axis=1)
        self.total_sales_val = self.chipo.apply(lambda row:row['quantity'] * row['item_price'], axis=1).sum()
        return self.total_sales_val
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        return len(self.chipo['order_id'].unique())
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        return round(self.total_sales_val/self.num_orders(),2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return len(self.chipo['item_name'].unique())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        df = pd.DataFrame(letter_counter.items(), columns=['item_name', 'frequency'])
        # 2. sort the values from the top to the least value and slice the first 5 items
        df = df.sort_values('frequency',ascending=False).head(5)
        # 3. create a 'bar' plot from the DataFrame
        bar_plot = df.plot.bar(x='item_name', y='frequency', rot=0)
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        
        # 5. show the plot. Hint: plt.show(block=True).
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    #assert order_id == 713926	
    assert quantity == 761 #The actual value is 761
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    
use pav_regression::{IsotonicRegression, Point};

fn main() {
    // Example: A retailer wants to understand the relationship between price and sales
    // Lower prices generally lead to more sales (descending relationship)
    let data = vec![
        Point::new(10.0, 100.0), // At $10, we sell 100 units
        Point::new(15.0, 80.0),  // At $15, we sell 80 units
        Point::new(20.0, 85.0),  // At $20, we sell 85 units (noise in data)
        Point::new(25.0, 60.0),  // At $25, we sell 60 units
        Point::new(30.0, 40.0),  // At $30, we sell 40 units
    ];

    let regression = IsotonicRegression::new_descending(&data).unwrap();

    println!("Price-to-Sales Prediction (interpolate):");
    println!("=========================================");

    // Use interpolate to predict sales at different prices
    for price in [12.0, 17.5, 22.5, 27.5] {
        let predicted_sales = regression.interpolate(price).unwrap();
        println!("  At price ${:.2}, we predict {:.1} sales", price, predicted_sales);
    }

    println!("\nSales-to-Price Prediction (invert):");
    println!("====================================");

    // Use invert to find what price will give us target sales
    for target_sales in [90.0, 75.0, 65.0, 50.0] {
        let required_price = regression.invert(target_sales).unwrap();
        println!(
            "  To sell {:.0} units, set price at ${:.2}",
            target_sales, required_price
        );
    }

    println!("\nRound-trip test (interpolate -> invert):");
    println!("=========================================");

    // Verify that invert and interpolate are true inverses
    let test_price = 18.0;
    let sales = regression.interpolate(test_price).unwrap();
    let price_back = regression.invert(sales).unwrap();
    println!("  Price ${:.2} -> {:.1} sales -> ${:.2} price", test_price, sales, price_back);
    println!("  Difference: ${:.4}", (test_price - price_back).abs());
}

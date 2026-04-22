import express from 'express'; // Import the Express framework
import path from 'path'; // Import the 'path' module to handle file paths
import fs from 'fs'; // Import the 'fs' module to read files
import { fileURLToPath } from 'url'; // Import fileURLToPath to get the file path from URL
import Papa from 'papaparse'; // Import PapaParse for parsing CSV files
import cors from 'cors'; // Import CORS middleware for enabling Cross-Origin Resource Sharing

const app = express(); // Create an instance of an Express application

// Get the current file's path and directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Enable CORS for all routes
app.use(cors());

// Define a route to handle GET requests to '/api/data'
app.get('/api/data', (req, res) => {
  // Define the path to the CSV file
  const csvFilePath = path.join(__dirname, 'data.csv');

  // Read the CSV file
  fs.readFile(csvFilePath, 'utf8', (err, data) => {
    if (err) {
      // Handle errors in reading the file
      console.error('Error reading CSV file:', err);
      return res.status(500).send('Error reading CSV file'); // Respond with a 500 status code
    }

    // Parse the CSV data
    Papa.parse(data, {
      header: true, // Specify that the CSV has headers
      complete: (results) => {
        // Send the parsed data as a JSON response
        res.json(results.data);
      }
    });
  });
});

// Define the port to listen on, default to 5001 if not specified
const PORT = process.env.PORT || 5001;

// Start the server and listen on the specified port
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

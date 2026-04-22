import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

//To set up the React application and render the main App component into the root DOM node.

ReactDOM.createRoot(document.getElementById('root')).render(  //creates a root to the React application at the DOM element ID:root
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

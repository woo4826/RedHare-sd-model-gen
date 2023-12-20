# Use the official Node.js image with Alpine Linux
FROM node:16-alpine3.11

# Set the working directory inside the container
WORKDIR /blog

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install dependencies
RUN npm install --force

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Install serve globally
RUN npm install -g serve

# Expose port 3000 (or the port your application uses)
EXPOSE 3001

# Command to run when the container starts
CMD [ "serve", "-s", "build" ]

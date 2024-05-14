// Function to fetch data from Flask endpoint
function fetchYearlyData() {
    fetch('/dashboard', {
        method: 'POST'  // Use POST method to trigger data processing in Flask
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received yearly data:', data);
        // Call function to create plot using received data
        createYearlyPlot(data);
    })
    .catch(error => console.error('Error fetching yearly data:', error));
}

// Function to create plot using D3.js for yearly data
function createYearlyPlot(data) {
    // Extract yearly data from received data
    const yearlyData = data.yearly;

    // D3.js code for creating the plot
    const svgWidth = 640;
    const svgHeight = 400;
    const margin = { top: 50, right: 20, bottom: 20, left: 80 };
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    // Append SVG to the DOM
    const svg = d3.select('#chart3')
        .append('svg')
        .attr('width', svgWidth)
        .attr('height', svgHeight);

    // Create scales for x and y axes
    const xScale = d3.scaleLinear()
        .domain([0, yearlyData.length - 1])
        .range([margin.left, width + margin.left]);

    const yScale = d3.scaleLinear()
        .domain([10000, d3.max(yearlyData, d => d.AEP_MW)])
        .range([height + margin.top, margin.top]);

    // Create line function
    const line = d3.line()
        .x((d, i) => xScale(i))
        .y(d => yScale(d.AEP_MW));

    // Append line to SVG
// Append line to SVG
const linePath = svg.append('path')
    .datum(yearlyData)
    .attr('fill', 'none')
    .attr('stroke', '#CC3333')
    .attr('stroke-width', 2)
    .attr('d', line)
    .on('mouseover', function() {
        d3.select(this)
            .attr('stroke', 'orange'); // Change color on mouseover
    })
    .on('mouseout', function() {
        d3.select(this)
            .attr('stroke', '#CC3333'); // Revert to original color on mouseout
    });

// Transition for line
linePath.transition()
    .duration(2000) 
    .attr('stroke-dasharray', function () {
        const length = this.getTotalLength();
        return length + ' ' + length;
    })
    .attr('stroke-dashoffset', function () {
        return this.getTotalLength();
    })
    .transition()
    .duration(6000) // Transition duration in milliseconds
    .ease(d3.easeLinear)
    .attr('stroke-dashoffset', 0);

    // Append x axis
    svg.append('g')
        .attr('transform', `translate(0,${height + margin.top})`)
        .call(d3.axisBottom(xScale));

    // Append y axis
    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale));

    // Append plot title
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", margin.top / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "24px")
        .style("fill", "white")
        .text("Energy consumption (Yearly)");
}

// Call fetchYearlyData function when the page loads to retrieve yearly data and create the plot
document.addEventListener('DOMContentLoaded', () => {
    fetchYearlyData();
});

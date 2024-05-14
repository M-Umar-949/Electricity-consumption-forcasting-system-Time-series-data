        // Sample monthly data (replace this with actual fetched data)
        const Data = [
            { month: 'January', consumption: 15000 },
            { month: 'February', consumption: 18000 },
            { month: 'March', consumption: 21000 },
            { month: 'April', consumption: 19000 },
            { month: 'May', consumption: 20000 },
            { month: 'June', consumption: 22000 },
            { month: 'July', consumption: 25000 },
            { month: 'August', consumption: 24000 },
            { month: 'September', consumption: 23000 },
            { month: 'October', consumption: 21000 },
            { month: 'November', consumption: 18000 },
            { month: 'December', consumption: 16000 }
        ];
        
        // D3 code to create a bar chart
        const svgWidth = 640;
        const svgHeight = 400;
        const margin = { top: 50, right: 20, bottom: 50, left: 80 };
        const width = svgWidth - margin.left - margin.right;
        const height = svgHeight - margin.top - margin.bottom;

        const svg = d3.select('#chart4')
            .append('svg')
            .attr('width', svgWidth)
            .attr('height', svgHeight);

        const xScale = d3.scaleBand()
            .domain(Data.map(d => d.month))
            .range([margin.left, width + margin.left])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(Data, d => d.consumption)])
            .nice()
            .range([height + margin.top, margin.top]);

            svg.selectAll('rect')
            .data(Data)
            .enter()
            .append('rect')
            .attr('x', d => xScale(d.month))
            .attr('y', height + margin.top) // Start the bars from the bottom
            .attr('width', xScale.bandwidth())
            .attr('height', 0) // Set initial height to 0
            .attr('fill', 'white')
            // .attr('opacity', 0.5)

            .on('mouseover', function () {
                d3.select(this).attr('fill', '#FF4545'); // Change color on mouseover
            })
            .on('mouseout', function () {
                d3.select(this).attr('fill', 'white'); // Change color back on mouseout
            })
            .transition() // Add transition
            .duration(6000) // Transition duration in milliseconds
            .attr('y', d => yScale(d.consumption)) // Move bars to correct position vertically
            .attr('height', d => height + margin.top - yScale(d.consumption));
        

        svg.append('g')
            .attr('transform', `translate(0, ${height + margin.top})`)
            .call(d3.axisBottom(xScale))
            .selectAll('text')
            .style('text-anchor', 'end')
            .attr('transform', 'rotate(-45)');

        svg.append('g')
            .attr('transform', `translate(${margin.left}, 0)`)
            .call(d3.axisLeft(yScale));



        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -svgHeight / 2)
            .attr('y', margin.left / 2)
            .attr('text-anchor', 'middle')
            .text('Energy Consumption');

            svg.append("text")
            .attr("x", width / 2)
            .attr("y", margin.top / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "24px")
            .style("fill", "white")
            .text("Average consumption (monthly)");
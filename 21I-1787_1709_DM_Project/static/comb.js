d3.csv('static/combinedforecast.csv').then((data) => {
    // Convert the data to a format that can be used by D3
    const formattedData = data.map((d) => {
      return {
        datetime: new Date(d.Datetime),
        ARIMA_Forecast: +d.ARIMA_Forecast,
        SARIMA_Forecast: +d.SARIMA_Forecast,
        ETS_Forecast: +d.ETS_Forecast,
        ANN_Forecast: +d.ANN_Forecast,
        LSTM_Forecast: +d.LSTM_Forecast,
        HYBRID_Forecast: +d.HYBRID_Forecast,
      };
    });

    const legendData = [
    { color: 'steelblue', text: 'ARIMA Forecast' },
    { color: 'red', text: 'SARIMA Forecast' },
    { color: 'green', text: 'ETS Forecast' },
    { color: 'orange', text: 'ANN Forecast' },
    { color: 'purple', text: 'LSTM Forecast' },
    { color: 'pink', text: 'HYBRID Forecast' }
  ];

    // Set up the SVG
    const margin = { top: 50, right: 20, bottom: 20, left: 80 };
    const width = 640 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const opacity = 0.4;

    const svg = d3
      .select('#chart1')
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Set up the scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(formattedData, (d) => d.datetime))
      .range([0, width]);

    const yScale = d3
      .scaleLinear()
      .domain([14000, d3.max(formattedData, (d) => Math.max(d.ARIMA_Forecast, d.SARIMA_Forecast, d.ETS_Forecast, d.ANN_Forecast, d.LSTM_Forecast, d.HYBRID_Forecast))])
      .range([height, 0]);

    // Add the area charts


    const areaGenerator2 = d3
      .area()
      .x((d) => xScale(d.datetime))
      .y0(height)
      .y1((d) => yScale(d.SARIMA_Forecast));

    svg
      .append('path')
      .datum(formattedData)
      .attr('class', 'area')
      .attr('fill', 'red')
      .attr('opacity', opacity)
      .attr('d', areaGenerator2);

    const areaGenerator3 = d3
      .area()
      .x((d) => xScale(d.datetime))
      .y0(height)
      .y1((d) => yScale(d.ETS_Forecast));

    svg
      .append('path')
      .datum(formattedData)
      .attr('class', 'area')
      .attr('fill', 'green')
      .attr('opacity', opacity)
      .attr('d', areaGenerator3);

    const areaGenerator4 = d3
      .area()
      .x((d) => xScale(d.datetime))
      .y0(height)
      .y1((d) => yScale(d.ANN_Forecast));

    svg
      .append('path')
      .datum(formattedData)
      .attr('class', 'area')
      .attr('fill', 'orange')
      .attr('opacity', opacity)
      .attr('d', areaGenerator4);

    const areaGenerator5 = d3
      .area()
      .x((d) => xScale(d.datetime))
      .y0(height)
      .y1((d) => yScale(d.LSTM_Forecast));

    svg
      .append('path')
      .datum(formattedData)
      .attr('class', 'area')
      .attr('fill', 'purple')
      .attr('opacity', opacity)
      .attr('d', areaGenerator5);

    const areaGenerator6 = d3
      .area()
      .x((d) => xScale(d.datetime))
      .y0(height)
      .y1((d) => yScale(d.HYBRID_Forecast));

      svg
      .append('path')
      .datum(formattedData)
      .attr('class', 'area')
      .attr('fill', 'pink')
      .attr('opacity', opacity)
      .attr('d', areaGenerator6);

      const areaGenerator = d3
      .area()
      .x((d) => xScale(d.datetime))
      .y0(height)
      .y1((d) => yScale(d.ARIMA_Forecast));

    svg
      .append('path')
      .datum(formattedData)
      .attr('class', 'area')
      .attr('fill', 'steelblue')
      .attr('opacity', opacity)
      .attr('d', areaGenerator);

      svg.selectAll('.area')
      .on('mouseover', function (event, d) {
        d3.select(this).attr('opacity', 0,7); // Increase opacity on mouseover
      })
      .on('mouseout', function (d) {
        d3.select(this).attr('opacity', opacity); // Revert to original opacity on mouseout
      });

    svg
      .append('g')
      .attr('transform', `translate(0, ${height})`)
      .call(d3.axisBottom(xScale));

    svg
      .append('g')
      .call(d3.axisLeft(yScale));

      const legend = svg.selectAll('.legend')
      .data(legendData)
      .enter().append('g')
      .attr('class', 'legend')
      .attr('transform', (d, i) => `translate(0, ${i * 20})`); // Adjust spacing here

    legend.append('rect')
      .attr('x', width-400)
      .attr('y', 200)
      .attr('width', 18)
      .attr('height', 18)
      .style('fill', (d) => d.color)
      .attr('opacity',opacity);

    legend.append('text')
      .attr('x', width -406)
      .attr('y', 210)
      .attr('dy', '.25em')
      .style('text-anchor', 'end')
      .attr('fill','White')
      .style('font-size', '12px')
      .text((d) => d.text);

      svg.append("text")
      .attr("x", width / 2 -50)
      .attr("y", margin.top / 2-50)
      .attr("text-anchor", "middle")
      .style("font-size", "24px")
      .style("fill", "white")
      .text("Comparison of 12 hours from all models");

          svg.selectAll('.area')
      .on('mouseover', function(event, d) {
        // Transition for pop-up effect
        d3.select(this)
          .transition()
          .duration(200)
          .attr('opacity', 0.7) // Adjust opacity on hover for pop-up effect
          .attr('transform', 'scale(1.1)'); // Example: Increase size

        // Add transition delays or other effects as needed

      })
      .on('mouseout', function(d) {
        // Transition to revert to original state
        d3.select(this)
          .transition()
          .duration(500)
          .attr('opacity', opacity) // Revert opacity on mouseout
          .attr('transform', 'scale(1)'); // Example: Revert size

        // Add transition delays or other effects as needed

      });

  });

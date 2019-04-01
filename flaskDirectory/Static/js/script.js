function suicideRate() {
    this.init()
}

suicideRate.prototype = {
    init() {
        d3.select('#scree').remove();
        $('.loader').hide();
        $('.overlay').hide();

        d3.select("#plotter")
            .on("change", this.getDataToShow);
        d3.select("#sampler")
            .on("change", this.getDataToShow);
        d3.select("#screeCheck")
            .on("change", this.getDataToShow);
        d3.select("#highestCheck")
            .on("change", this.getDataToShow);

        this.getDataToShow();
    },
    getDataToShow() {
        var plotter = $('#plotter').val();
        var isMatrix = false;
        var isScree = false;
        var chartTitle = '';

        var sampler = $('#sampler').val();
        var isRandomSampling = false;
        if (sampler == 'rand') {
            isRandomSampling = true;
        }
        if (plotter == 'scat_mat') {
            isMatrix = true;
            chartTitle = 'Scatter matrix';
        } else if (plotter == 'pca') {
            chartTitle = 'PCA'
        } else if (plotter == 'mds_cor') {
            chartTitle = 'MDS Correlation'
        } else if (plotter == 'scree') {
            chartTitle = 'Scree Plot';
            isScree = true;
            isRandomSampling = false;
            isMatrix = false;
        } else {
            chartTitle = 'MDS Euclidian'
        }
        isOriginal = $("[name='checkboxOriginal']").prop('checked')
        if (isOriginal){
            chartTitle = 'Original Data Scree Plot';
            isScree = true;
            isRandomSampling = false;
            isMatrix = false;
        }
        isHighest = $("[name='highestCheck']").prop('checked')
        if(isHighest){
            chartTitle = '3 Highest PCA Scree Plot';
            isScree = true;
            isRandomSampling = false;
            isMatrix = false;
        }

        get_map('/display_plots', isRandomSampling, isMatrix, isScree,isOriginal, isHighest, chartTitle);
    }
}

function get_map(url, randomSamples, matrix, isScree,isOriginal,isHighest,chart_title) {
    $('.overlay').show();
    $('.loader').show();
    var showScreeOriginal = 0
    var showHighest = 0
    if(isOriginal){
        showScreeOriginal = 1
    }
    if(isHighest){
        showScreeOriginal  =1;
        showHighest =1
    }

    $.getJSON($SCRIPT_ROOT + url, {
        plotter: $('#plotter').val(),
        sampler: $('#sampler').val(),
        scree: showScreeOriginal,
        highest : showHighest
    }, function (result) {
        $('.overlay').hide();
        $('.loader').hide();
        console.log(result)

        if (matrix) {
            drawScatterPlotMatrix(result, randomSamples, chart_title);
        }
        else {
            drawScatter(result, randomSamples, chart_title);
        }
        if (isScree) {
            draw_scree_plot(result, chart_title)
        }
    });
}

function draw_scree_plot(eigen_values, chart_title) {

    var data = eigen_values;
    d3.select('#chart').remove();

    var margin = {top: 20, right: 20, bottom: 30, left: 60};
    var width = 1366 - margin.left - margin.right;
    var height = 450 - margin.top - margin.bottom;

    var chart_width = 800;
    var chart_height = height + margin.top + margin.bottom;

    var x = d3.scaleLinear().domain([1, data.length + 0.5]).range([0, chart_width - 120]);
    var y = d3.scaleLinear().domain([0, d3.max(data)]).range([height, 0]);

    var xAxis = d3.axisBottom(x);
    var yAxis = d3.axisLeft(y);

    var markerX
    var markerY
    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var line = d3.line()
        .x(function (d, i) {
            if (i == 3) {
                markerX = x(i);
                markerY = y(d)
            }
            return x(i);
        })
        .y(function (d) {
            return y(d);
        })

    var graph = d3.select("body").append("svg")
        .attr("id", "chart")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom + 10)
        .append("g")
        .attr("transform", "translate(250,10)");

    graph.append("g")
        .attr("class", "x_axis")
        .attr("transform", "translate(110," + height + ")")
        .call(xAxis);

    graph.append("g")
        .attr("class", "y_axis")
        .attr("transform", "translate(100,0)")
        .call(yAxis);

    graph.append("path")
        .attr("d", line(data))
        .attr("transform", "translate(215,0)")
        .attr("fill", "none")
        .attr("stroke", color(1))
        .attr("stroke-width", "3px")

    graph.append("circle")
        .attr("cx", markerX)
        .attr("cy", markerY)
        .attr("r", 8)
        .attr("transform", "translate(215,0)")
        .style("fill", "red");

    graph.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + (50) + "," + (height / 2) + ")rotate(-90)")
        .text("Eigen Values");

    graph.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + (chart_width / 2) + "," + (chart_height) + ")")
        .text("K");

    graph.append("text")
        .attr("x", (width / 3))
        .attr("y", 0 + (margin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .text(chart_title);
}

function drawScatterPlotMatrix(sData, randomSamples, chart_title) {
    d3.select('#chart').remove();

    var jdata = sData
    var attributeNames = Object.keys(jdata);
    var width = 1020,
        size = 230,
        padding = 30;

    var x = d3.scaleLinear()
        .range([padding / 2, size - padding / 2]);

    var y = d3.scaleLinear()
        .range([size - padding / 2, padding / 2]);

    var xAxis = d3.axisBottom(x).ticks(6);

    var yAxis = d3.axisLeft(y).ticks(6);

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    data = {};
    data[attributeNames[0]] = jdata[attributeNames[0]];
    data[attributeNames[1]] = jdata[attributeNames[1]];
    data[attributeNames[2]] = jdata[attributeNames[2]];
    data[attributeNames[3]] = jdata[attributeNames[3]];


    var domainByFtr = {},
        attributeNames = d3.keys(data).filter(function (d) {
            return d !== "clusterid";
        }),
        n = attributeNames.length;

    xAxis.tickSize(size * n);
    yAxis.tickSize(-size * n);

    attributeNames.forEach(function (ftrName) {
        domainByFtr[ftrName] = d3.extent(d3.values(data[ftrName]));
    });

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("width", size * n + padding)
        .attr("height", size * n + padding)
        .append("g")
        .attr("transform", "translate(" + padding + "," + padding / 2 + ")");

    svg.selectAll(".x.axis")
        .data(attributeNames)
        .enter().append("g")
        .attr("class", "x axis")
        .attr("transform", function (d, i) {
            return "translate(" + (n - i - 1) * size + ",0)";
        })
        .each(function (d) {
            x.domain(domainByFtr[d]);
            d3.select(this).call(xAxis);
        });

    svg.selectAll(".y.axis")
        .data(attributeNames)
        .enter().append("g")
        .attr("class", "y axis")
        .attr("transform", function (d, i) {
            return "translate(0," + i * size + ")";
        })
        .each(function (d) {
            y.domain(domainByFtr[d]);
            d3.select(this).call(yAxis);
        });

    svg.append("text")
        .attr("x", (width / 2.8))
        .attr("y", 0 + (5))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .text(chart_title);

    var cell = svg.selectAll(".cell")
        .data(cross(attributeNames, attributeNames))
        .enter().append("g")
        .attr("class", "cell")
        .attr("transform", function (d) {
            return "translate(" + (n - d.i - 1) * size + "," + d.j * size + ")";
        })
        .each(plot);

    cell.filter(function (d) {
        return d.i === d.j;
    }).append("text")
        .attr("x", padding)
        .attr("y", padding)
        .attr("dy", ".71em")
        .text(function (d) {
            return d.x;
        });

    function plot(p) {
        var cell = d3.select(this);
        x.domain(domainByFtr[String(p.x)]);
        y.domain(domainByFtr[String(p.y)]);
        cell.append("rect")
            .attr("class", "frame")
            .attr("x", padding / 2)
            .attr("y", padding / 2)
            .attr("width", size - padding)
            .attr("height", size - padding);

        first_comp = data[String(p.x)];
        second_comp = data[String(p.y)];
        result_array = []
        second = d3.values(second_comp)
        cluster = data['clusterid']
        d3.values(first_comp).forEach(function (item, index) {
            tempMap = {};
            tempMap["x"] = item;
            tempMap["y"] = second[index];
            tempMap["clusterid"] = cluster[index];
            result_array.push(tempMap);
        });

        cell.selectAll("circle")
            .data(result_array)
            .enter().append("circle")
            .attr("cx", function (d) {
                return x(d.x);
            })
            .attr("cy", function (d) {
                return y(d.y);
            })
            .attr("r", 4)
            .style("fill", function (d) {
                return randomSamples ? color("orange") : color(d.clusterid);
            });
    }
}

function cross(a, b) {
    var c = [], n = a.length, m = b.length, i, j;
    for (i = -1; ++i < n;) for (j = -1; ++j < m;) c.push({x: a[i], i: i, y: b[j], j: j});
    return c;
}

function drawScatter(sData, randomSamples, chart_title) {
    d3.select('#chart').remove();
    var data = sData;
    var array = [];
    var min = 0, max = 0;
    attributeNames = Object.keys(data);
    for (var i = 0; i < Object.keys(data[0]).length; ++i) {
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];
        obj.clusterid = data['clusterid'][i]
        obj.ftr1 = data[attributeNames[2]][i]
        obj.ftr2 = data[attributeNames[3]][i]
        array.push(obj);
    }
    data = array;

    var margin = {top: 20, right: 20, bottom: 30, left: 40},
        width = 1080 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    var xValue = function (d) {
            return d.x;
        }, xScale = d3.scaleLinear().range([0, width]),
        xMap = function (d) {
            return xScale(xValue(d));
        }, xAxis = d3.axisBottom(xScale);

    var yValue = function (d) {
            return d.y;
        }, yScale = d3.scaleLinear().range([height, 0]),
        yMap = function (d) {
            return yScale(yValue(d));
        }, yAxis = d3.axisLeft(yScale);

    var cValue
    if (randomSamples) {
        cValue = function (d) {
            return d.clusteridx;
        }
    } else {
        cValue = function (d) {
            return d.clusterid;
        }
    }
    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var tooltip = d3.select("body").append('div').style('position', 'absolute');

    xScale.domain([d3.min(data, xValue) - 1, d3.max(data, xValue) + 1]);
    yScale.domain([d3.min(data, yValue) - 1, d3.max(data, yValue) + 1]);

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .attr("class", "x_axis")
        .call(xAxis)
        .append("text")
        .attr("class", "label")
        .attr("y", -6)
        .attr("x", width)
        .text("Component 1")
        .style("text-anchor", "end");

    svg.append("g")
        .attr("class", "y_axis")
        .call(yAxis)
        .append("text")
        .attr("class", "label")
        .attr("y", 6)
        .attr("transform", "rotate(-90)")
        .attr("dy", ".71em")
        .text("Component 2")
        .style("text-anchor", "end");

    svg.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", xMap)
        .attr("r", 3.5)
        .attr("cy", yMap)
        .style("fill", function (d) {
            return color(cValue(d));
        })
        .on("mouseover", function (d) {
            if(randomSamples){
                d3.select(this).style("fill", 'red')
            }

            tooltip.transition().style('opacity', 1).style('color', 'black')
            tooltip.html(attributeNames[2] + " : " + d.ftr1 + ", " + attributeNames[3] + " : " + d.ftr2)
                .style("top", (d3.event.pageY - 28) + "px")
                .style("left", (d3.event.pageX + 5) + "px");
        })
        .on("mouseout", function (d) {
            if(randomSamples){
                d3.select(this).style("fill", 'steelblue')
            }

            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
            tooltip.html('');
        });

    svg.append("text")
        .attr("x", 700)
        .attr("y", 0 + (margin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("font-weight", "bold")
        .text(chart_title);
}
